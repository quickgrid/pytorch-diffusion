"""Implementation of DDPM.

Best to use corrupted, low res image mover script first then use batch image resizer to resize image
to expected format before using this.

References
    - DDPM paper, https://arxiv.org/pdf/2006.11239.pdf.
    - DDIM paper, https://arxiv.org/pdf/2010.02502.pdf.
    - Annotated Diffusion, https://huggingface.co/blog/annotated-diffusion.
    - Keras DDIM, https://keras.io/examples/generative/ddim/.
    - Implementation, https://www.youtube.com/watch?v=TBCRlnwJtZU.
    - Implementation, https://github.com/dome272/Diffusion-Models-pytorch.
    - Postional embedding, http://nlp.seas.harvard.edu/annotated-transformer/.
    - Attention paper, https://arxiv.org/pdf/1706.03762.pdf.
    - Transformers, https://pytorch.org/tutorials/beginner/transformer_tutorial.html.
    - Transformer encoder architecture, https://arxiv.org/pdf/2010.11929.pdf.
    - UNet architecture, https://arxiv.org/pdf/1505.04597.pdf.
"""
import copy
import math
import os
import logging
import pathlib
from typing import Tuple

import torch
import torch.nn as nn
import torchvision.utils
from PIL import Image
from torch.cuda.amp import GradScaler
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm
from torch import optim
from torch.functional import F
from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


class Diffusion:
    def __init__(
            self,
            device: str,
            img_size: int,
            noise_steps: int = 1000,
            beta_start: float = 1e-4,
            beta_end: float = 0.02,
    ):
        self.device = device
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size

        # Section 2, equation 4 and near explation for alpha, alpha hat, beta.
        self.beta = self.linear_noise_schedule()
        self.alpha = 1 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        # Section 3.2, algorithm 1 formula implementation. Generate values early reuse later.
        self.sqrt_alpha_hat = torch.sqrt(self.alpha_hat)
        self.sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat)

        # Section 3.2, equation 2 precalculation values.
        self.sqrt_alpha = torch.sqrt(self.alpha)
        self.std_beta = torch.sqrt(self.beta)

        # Clean up unnecessary values.
        del self.alpha
        del self.alpha_hat

    def linear_noise_schedule(self) -> torch.Tensor:
        """Same amount of noise is applied each step. Weakness is near end steps image is so noisy it is hard make
        out information. So noise removal is also very small amount, so it takes more steps to generate clear image.
        """
        return torch.linspace(start=self.beta_start, end=self.beta_end, steps=self.noise_steps, device=self.device)

    def q_sample(self, x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Section 3.2, algorithm 1 formula implementation. Forward process, defined by `q`.

        Found in section 2. `q` gradually adds gaussian noise according to variance schedule. Also,
        can be seen on figure 2.
        """
        sqrt_alpha_hat = self.sqrt_alpha_hat[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_hat = self.sqrt_one_minus_alpha_hat[t].view(-1, 1, 1, 1)
        epsilon = torch.randn_like(x, device=self.device)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon, epsilon

    def sample_timesteps(self, batch_size: int) -> torch.Tensor:
        """Random timestep for each sample in a batch. Timesteps selected from [1, noise_steps].
        """
        return torch.randint(low=1, high=self.noise_steps, size=(batch_size, ), device=self.device)

    def p_sample(self, eps_model: nn.Module, n: int, scale_factor: int = 2) -> torch.Tensor:
        """Implementation of algorithm 2 sampling. Reverse process, defined by `p` in section 2. Short
         formula is defined in equation 11 of section 3.2.

        From noise generates image step by step. From noise_steps, (noise_steps - 1), ...., 2, 1.
        Here, alpha = 1 - beta. So, beta = 1 - alpha.

        Sample noise from normal distribution of timestep t > 1, else noise is 0. Before returning values
        are clamped to [-1, 1] and converted to pixel values [0, 255].

        Args:
            scale_factor: Scales the output image by the factor.
            eps_model: Noise prediction model. `eps_theta(x_t, t)` in paper. Theta is the model parameters.
            n: Number of samples to process.

        Returns:
            Generated denoised image.
        """
        logging.info(f'Sampling {n} images....')

        eps_model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size), device=self.device)

            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = torch.ones(n, dtype=torch.long, device=self.device) * i

                sqrt_alpha_t = self.sqrt_alpha[t].view(-1, 1, 1, 1)
                beta_t = self.beta[t].view(-1, 1, 1, 1)
                sqrt_one_minus_alpha_hat_t = self.sqrt_one_minus_alpha_hat[t].view(-1, 1, 1, 1)
                epsilon_t = self.std_beta[t].view(-1, 1, 1, 1)

                random_noise = torch.randn_like(x) if i > 1 else torch.zeros_like(x)

                x = ((1 / sqrt_alpha_t) * (x - ((beta_t / sqrt_one_minus_alpha_hat_t) * eps_model(x, t)))) +\
                    (epsilon_t * random_noise)

        eps_model.train()

        x = ((x.clamp(-1, 1) + 1) * 127.5).type(torch.uint8)
        x = F.interpolate(input=x, scale_factor=scale_factor, mode='nearest-exact')
        return x

    def generate_gif(
            self,
            eps_model: nn.Module,
            n: int = 1,
            save_path: str = '',
            output_name: str = None,
            skip_steps: int = 20,
            scale_factor: int = 2,
    ) -> None:
        logging.info(f'Generating gif....')
        frames_list = []

        eps_model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size), device=self.device)

            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = torch.ones(n, dtype=torch.long, device=self.device) * i

                sqrt_alpha_t = self.sqrt_alpha[t].view(-1, 1, 1, 1)
                beta_t = self.beta[t].view(-1, 1, 1, 1)
                sqrt_one_minus_alpha_hat_t = self.sqrt_one_minus_alpha_hat[t].view(-1, 1, 1, 1)
                epsilon_t = self.std_beta[t].view(-1, 1, 1, 1)

                random_noise = torch.randn_like(x) if i > 1 else torch.zeros_like(x)

                x = ((1 / sqrt_alpha_t) * (x - ((beta_t / sqrt_one_minus_alpha_hat_t) * eps_model(x, t)))) +\
                    (epsilon_t * random_noise)

                if i % skip_steps == 0:
                    x_img = F.interpolate(input=x, scale_factor=scale_factor, mode='nearest-exact')
                    x_img = ((x_img.clamp(-1, 1) + 1) * 127.5).type(torch.uint8)
                    grid = torchvision.utils.make_grid(x_img)
                    img_arr = grid.permute(1, 2, 0).cpu().numpy()
                    img = Image.fromarray(img_arr)
                    frames_list.append(img)

        eps_model.train()

        output_name = output_name if output_name else 'output'
        frames_list[0].save(
            os.path.join(save_path, f'{output_name}.gif'),
            save_all=True,
            append_images=frames_list[1:],
            optimize=False,
            duration=80,
            loop=0
        )


class PositionalEncoding(nn.Module):
    def __init__(
            self,
            embedding_dim: int,
            dropout: float = 0.1,
            max_len: int = 1000,
            apply_dropout: bool = True,
    ):
        """Section 3.5 of attention is all you need paper.

        Extended slicing method is used to fill even and odd position of sin, cos with increment of 2.
        Ex, `[sin, cos, sin, cos, sin, cos]` for `embedding_dim = 6`.

        `max_len` is equivalent to number of noise steps or patches. `embedding_dim` must same as image
        embedding dimension of the model.

        Args:
            embedding_dim: `d_model` in given positional encoding formula.
            dropout: Dropout amount.
            max_len: Number of embeddings to generate. Here, equivalent to total noise steps.
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.apply_dropout = apply_dropout

        pos_encoding = torch.zeros(max_len, embedding_dim)
        position = torch.arange(start=0, end=max_len).unsqueeze(1)
        div_term = torch.exp(-math.log(10000.0) * torch.arange(0, embedding_dim, 2).float() / embedding_dim)

        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer(name='pos_encoding', tensor=pos_encoding, persistent=False)

    def forward_on_all(self, x: torch.Tensor) -> torch.Tensor:
        """Adds positional embedding to all patches. Not used here.

        Same positional embedding is applied to all images in batch. The precalculated positional
        embedding values are summed with the given embedding vector.

        Args:
            x: Image patch embedding tensor. Shape [batch_size, patch_count, embedding_dim].
        """
        pos_encoding = self.pos_encoding.unsqueeze(0)
        x = x + pos_encoding[:, : x.shape[1]]
        return self.dropout(x)

    def forward(self, t: torch.LongTensor) -> torch.Tensor:
        """Get precalculated positional embedding at timestep t. Outputs same as video implementation
        code but embeddings are in [sin, cos, sin, cos] format instead of [sin, sin, cos, cos] in that code.
        Also batch dimension is added to final output.
        """
        positional_encoding = self.pos_encoding[t].squeeze(1)
        if self.apply_dropout:
            return self.dropout(positional_encoding)
        return positional_encoding


class DoubleConv(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            mid_channels: int = None,
            residual: bool = False
    ):
        """Double convolutions as applied in the unet paper architecture.
        """
        super(DoubleConv, self).__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels, out_channels=mid_channels, kernel_size=(3, 3), padding=(1, 1), bias=False
            ),
            nn.GroupNorm(num_groups=1, num_channels=mid_channels),
            nn.GELU(),
            nn.Conv2d(
                in_channels=mid_channels, out_channels=out_channels, kernel_size=(3, 3), padding=(1, 1), bias=False,
            ),
            nn.GroupNorm(num_groups=1, num_channels=out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.residual:
            return F.gelu(x + self.double_conv(x))

        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, emb_dim: int = 256):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2)),
            DoubleConv(in_channels=in_channels, out_channels=in_channels, residual=True),
            DoubleConv(in_channels=in_channels, out_channels=out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(in_features=emb_dim, out_features=out_channels),
        )

    def forward(self, x: torch.Tensor, t_embedding: torch.Tensor) -> torch.Tensor:
        """Downsamples input tensor, calculates embedding and adds embedding channel wise.

        If, `x.shape == [4, 64, 64, 64]` and `out_channels = 128`, then max_conv outputs [4, 128, 32, 32] by
        downsampling in h, w and outputting specified amount of feature maps/channels.

        `t_embedding` is embedding of timestep of shape [batch, time_dim]. It is passed through embedding layer
        to output channel dimentsion equivalent to channel dimension of x tensor, so they can be summbed elementwise.

        Since emb_layer output needs to be summed its output is also `emb.shape == [4, 128]`. It needs to be converted
        to 4D tensor, [4, 128, 1, 1]. Then the channel dimension is duplicated in all of `H x W` dimension to get
        shape of [4, 128, 32, 32]. 128D vector is sample for each pixel position is image. Now the emb_layer output
        is summed with max_conv output.
        """
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t_embedding)
        emb = emb.view(emb.shape[0], emb.shape[1], 1, 1).repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, emb_dim: int = 256):
        super(Up, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels=in_channels, out_channels=in_channels, residual=True),
            DoubleConv(in_channels=in_channels, out_channels=out_channels, mid_channels=in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(in_features=emb_dim, out_features=out_channels),
        )

    def forward(self, x: torch.Tensor, x_skip: torch.Tensor, t_embedding: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([x_skip, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t_embedding)
        emb = emb.view(emb.shape[0], emb.shape[1], 1, 1).repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class TransformerEncoderSA(nn.Module):
    def __init__(self, num_channels: int, size: int, num_heads: int = 4):
        """A block of transformer encoder with mutli head self attention from vision transformers paper,
         https://arxiv.org/pdf/2010.11929.pdf.
        """
        super(TransformerEncoderSA, self).__init__()
        self.num_channels = num_channels
        self.size = size
        self.mha = nn.MultiheadAttention(embed_dim=num_channels, num_heads=num_heads, batch_first=True)
        self.ln = nn.LayerNorm([num_channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([num_channels]),
            nn.Linear(in_features=num_channels, out_features=num_channels),
            nn.LayerNorm([num_channels]),
            nn.Linear(in_features=num_channels, out_features=num_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Self attention.

        Input feature map [4, 128, 32, 32], flattened to [4, 128, 32 x 32]. Which is reshaped to per pixel
        feature map order, [4, 1024, 128].

        Attention output is same shape as input feature map to multihead attention module which are added element wise.
        Before returning attention output is converted back input feature map x shape. Opposite of feature map to
        mha input is done which gives output [4, 128, 32, 32].
        """
        x = x.view(-1, self.num_channels, self.size * self.size).permute(0, 2, 1)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(query=x_ln, key=x_ln, value=x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.permute(0, 2, 1).view(-1, self.num_channels, self.size, self.size)


class UNet(nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            out_channels: int = 3,
            noise_steps: int = 1000,
            time_dim: int = 256,
            features: list = None,
    ):
        super(UNet, self).__init__()
        if features is None:
            features = [64, 128, 256, 512]
        self.time_dim = time_dim
        self.pos_encoding = PositionalEncoding(embedding_dim=time_dim, max_len=noise_steps)

        self.input_conv = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.sa1 = TransformerEncoderSA(128, 32)
        self.down2 = Down(128, 256)
        self.sa2 = TransformerEncoderSA(256, 16)
        self.down3 = Down(256, 256)
        self.sa3 = TransformerEncoderSA(256, 8)

        self.bottleneck1 = DoubleConv(256, 512)
        self.bottleneck2 = DoubleConv(512, 512)
        self.bottleneck3 = DoubleConv(512, 256)

        self.up1 = Up(512, 128)
        self.sa4 = TransformerEncoderSA(128, 16)
        self.up2 = Up(256, 64)
        self.sa5 = TransformerEncoderSA(64, 32)
        self.up3 = Up(128, 64)
        self.sa6 = TransformerEncoderSA(64, 64)
        self.out_conv = nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=(1, 1))

    def forward(self, x: torch.Tensor, t: torch.LongTensor) -> torch.Tensor:
        """Forward pass with image tensor and timestep reduce noise.

        Args:
            x: Image tensor of shape, [batch_size, channels, height, width].
            t: Time step defined as long integer. If batch size is 4 and timestep 100, then t = [100, 100, 100, 100].
        """
        t = self.pos_encoding(t)

        x1 = self.input_conv(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x4 = self.bottleneck1(x4)
        x4 = self.bottleneck2(x4)
        x4 = self.bottleneck3(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)

        return self.out_conv(x)


class EMA:
    def __init__(self, beta):
        """Modifies exponential moving average model.
        """
        self.beta = beta
        self.step = 0

    def update_model_average(self, ema_model: nn.Module, current_model: nn.Module) -> None:
        for current_params, ema_params in zip(current_model.parameters(), ema_model.parameters()):
            old_weights, new_weights = ema_params.data, current_params.data
            ema_params.data = self.update_average(old_weights=old_weights, new_weights=new_weights)

    def update_average(self, old_weights: torch.Tensor, new_weights: torch.Tensor) -> torch.Tensor:
        if old_weights is None:
            return new_weights
        return old_weights * self.beta + (1 - self.beta) * new_weights

    def ema_step(self, ema_model: nn.Module, model: nn.Module, step_start_ema: int = 2000) -> None:
        if self.step < step_start_ema:
            self.reset_parameters(ema_model=ema_model, model=model)
            self.step += 1
            return
        self.update_model_average(ema_model=ema_model, current_model=model)
        self.step += 1

    @staticmethod
    def reset_parameters(ema_model: nn.Module, model: nn.Module) -> None:
        ema_model.load_state_dict(model.state_dict())


class CustomImageClassDataset(Dataset):
    def __init__(
            self,
            root_dir: str,
            image_size: int,
            image_channels: int
    ):
        super(CustomImageClassDataset, self).__init__()
        self.root_dir = root_dir
        self.class_list = os.listdir(root_dir)

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5 for _ in range(image_channels)],
                std=[0.5 for _ in range(image_channels)],
            )
        ])

        self.image_labels_files_list = list()
        for idx, class_name_folder in enumerate(self.class_list):
            class_path = os.path.join(root_dir, class_name_folder)
            files_list = os.listdir(class_path)
            for image_file in files_list:
                self.image_labels_files_list.append(
                    (
                        os.path.join(class_path, image_file),
                        idx,
                    )
                )

        self.image_files_list_len = len(self.image_labels_files_list)

    def __len__(self) -> int:
        return self.image_files_list_len

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image_path, class_label = self.image_labels_files_list[idx]
        image = Image.open(image_path)
        image = image.convert('RGB')
        return self.transform(image), class_label


class Tester:
    def __init__(self, device: str = 'cuda', batch_size: int = 4, image_size: int = 64,):
        self.device = device
        self.batch_size = batch_size
        self.image_size = image_size

    def test_unet(self) -> None:
        net = UNet().to(self.device)
        print(f'Param count: {sum([p.numel() for p in net.parameters()])}')
        x = torch.randn(self.batch_size, 3, self.image_size, self.image_size)
        current_timestep = 500
        t = x.new_tensor([current_timestep] * x.shape[0]).long()
        output = net(x, t)
        assert x.shape == output.shape, 'Input image tensor and output image tensor of network should be same.'
        print(f'UNet input shape: {x.shape}')
        print(f'UNet output shape: {output.shape}')

    def test_attention(self) -> None:
        x = torch.randn(size=(4, 128, 32, 32))
        sa1 = TransformerEncoderSA(128, 32)
        output = sa1(x)
        assert x.shape == output.shape, 'Shape of output of feature map x and self attention output should be same.'
        print(f'Self attention input shape: {x.shape}')
        print(f'Self attention output shape: {output.shape}')

    def test_jit(self) -> None:
        net = torch.jit.script(UNet().to(self.device))
        print(f'Param count: {sum([p.numel() for p in net.parameters()])}')
        x = torch.randn(self.batch_size, 3, self.image_size, self.image_size)
        current_timestep = 500
        t = x.new_tensor([current_timestep] * x.shape[0]).long()
        output = net(x, t)
        assert x.shape == output.shape, 'Input image tensor and output image tensor of network should be same.'
        print(f'UNet input shape: {x.shape}')
        print(f'UNet output shape: {output.shape}')


class Utils:
    def __init__(self):
        super(Utils, self).__init__()

    @staticmethod
    def collate_fn(batch):
        """Discard none samples.
        """
        batch = list(filter(lambda x: x is not None, batch))
        return torch.utils.data.dataloader.default_collate(batch)

    @staticmethod
    def save_images(images: torch.Tensor, save_path: str) -> None:
        grid = torchvision.utils.make_grid(images)
        img_arr = grid.permute(1, 2, 0).cpu().numpy()
        img = Image.fromarray(img_arr)
        img.save(save_path)

    @staticmethod
    def save_checkpoint(
            epoch: int,
            model: nn.Module,
            filename: str,
            optimizer: optim.Optimizer = None,
            scheduler: optim.lr_scheduler = None,
            grad_scaler: GradScaler = None,
    ) -> None:
        checkpoint = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
        }
        if optimizer:
            checkpoint['optimizer'] = optimizer.state_dict()
        if scheduler:
            checkpoint['scheduler'] = scheduler.state_dict()
        if scheduler:
            checkpoint['grad_scaler'] = grad_scaler.state_dict()

        torch.save(checkpoint, filename)
        logging.info("=> Saving checkpoint complete.")

    @staticmethod
    def load_checkpoint(
            model: nn.Module,
            filename: str,
            optimizer: optim.Optimizer = None,
            scheduler: optim.lr_scheduler = None,
            grad_scaler: GradScaler = None,
    ) -> int:
        logging.info("=> Loading checkpoint")
        checkpoint = torch.load(filename, map_location="cuda")
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])
        if 'grad_scaler' in checkpoint:
            grad_scaler.load_state_dict(checkpoint['grad_scaler'])
        return checkpoint['epoch']


class Trainer:
    def __init__(
            self,
            dataset_path: str,
            save_path: str = None,
            checkpoint_path: str = None,
            checkpoint_path_ema: str = None,
            run_name: str = 'ddpm',
            image_size: int = 64,
            image_channels: int = 3,
            batch_size: int = 2,
            accumulation_iters: int = 16,
            sample_count: int = 1,
            num_workers: int = 0,
            device: str = 'cuda',
            num_epochs: int = 10000,
            fp16: bool = False,
            save_every: int = 2000,
            learning_rate: float = 2e-4,
            noise_steps: int = 500,
    ):
        self.num_epochs = num_epochs
        self.device = device
        self.fp16 = fp16
        self.save_every = save_every
        self.accumulation_iters = accumulation_iters
        self.sample_count = sample_count

        base_path = save_path if save_path is not None else os.getcwd()
        self.save_path = os.path.join(base_path, run_name)
        pathlib.Path(self.save_path).mkdir(parents=True, exist_ok=True)
        self.logger = SummaryWriter(log_dir=os.path.join(self.save_path, 'logs'))

        diffusion_dataset = CustomImageClassDataset(
            root_dir=dataset_path,
            image_size=image_size,
            image_channels=image_channels
        )
        self.train_loader = DataLoader(
            diffusion_dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=num_workers,
            drop_last=False,
            collate_fn=Utils.collate_fn,
        )

        self.unet_model = UNet().to(device)
        self.diffusion = Diffusion(img_size=image_size, device=self.device, noise_steps=noise_steps)
        self.optimizer = optim.Adam(
            params=self.unet_model.parameters(), lr=learning_rate,  # betas=(0.9, 0.999)
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer, T_max=300)
        # self.loss_fn = nn.MSELoss().to(self.device)
        self.grad_scaler = GradScaler()

        self.ema = EMA(beta=0.95)
        self.ema_model = copy.deepcopy(self.unet_model).eval().requires_grad_(False)

        # ema_avg = lambda avg_model_param, model_param, num_averaged: 0.1 * avg_model_param + 0.9 * model_param
        # self.swa_model = optim.swa_utils.AveragedModel(model=self.unet_model, avg_fn=ema_avg).to(self.device)
        # self.swa_start = 10
        # self.swa_scheduler = optim.swa_utils.SWALR(
        #     optimizer=self.optimizer, swa_lr=0.05, anneal_epochs=10, anneal_strategy='cos'
        # )

        self.start_epoch = 0
        if checkpoint_path:
            logging.info(f'Loading model weights...')
            self.start_epoch = Utils.load_checkpoint(
                model=self.unet_model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                grad_scaler=self.grad_scaler,
                filename=checkpoint_path,
            )
        if checkpoint_path_ema:
            logging.info(f'Loading EMA model weights...')
            _ = Utils.load_checkpoint(
                model=self.ema_model,
                filename=checkpoint_path_ema,
            )

    def sample(
            self,
            epoch: int = None,
            batch_idx: int = None,
            sample_count: int = 1,
            output_name: str = None
    ) -> None:
        """Generates images with reverse process based on sampling method with both training model and ema model.
        """
        sampled_images = self.diffusion.p_sample(eps_model=self.unet_model, n=sample_count)
        ema_sampled_images = self.diffusion.p_sample(eps_model=self.ema_model, n=sample_count)

        model_name = f'model_{epoch}_{batch_idx}.jpg'
        ema_model_name = f'model_ema_{epoch}_{batch_idx}.jpg'

        if output_name:
            model_name = f'{output_name}.jpg'
            ema_model_name = f'{output_name}_ema.jpg'

        Utils.save_images(
            images=sampled_images,
            save_path=os.path.join(self.save_path, model_name)
        )
        Utils.save_images(
            images=ema_sampled_images,
            save_path=os.path.join(self.save_path, ema_model_name)
        )

    def sample_gif(
            self,
            save_path: str = '',
            sample_count: int = 1,
            output_name: str = None,
    ) -> None:
        """Generates images with reverse process based on sampling method with both training model and ema model.
        """
        self.diffusion.generate_gif(
            eps_model=self.unet_model,
            n=sample_count,
            save_path=save_path,
            output_name=output_name,
        )
        self.diffusion.generate_gif(
            eps_model=self.ema_model,
            n=sample_count,
            save_path=save_path,
            output_name=f'{output_name}_ema',
        )

    def train(self) -> None:
        logging.info(f'Training started....')
        for epoch in range(self.start_epoch, self.num_epochs):
            # total_loss = 0.0
            accumulated_minibatch_loss = 0.0

            with tqdm(self.train_loader) as pbar:
                for batch_idx, (real_images, _) in enumerate(pbar):
                    real_images = real_images.to(self.device)
                    current_batch_size = real_images.shape[0]
                    t = self.diffusion.sample_timesteps(batch_size=current_batch_size)
                    x_t, noise = self.diffusion.q_sample(x=real_images, t=t)

                    with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=self.fp16):
                        predicted_noise = self.unet_model(x=x_t, t=t)

                        loss = F.smooth_l1_loss(predicted_noise, noise)
                        loss /= self.accumulation_iters

                        accumulated_minibatch_loss += float(loss)

                    self.grad_scaler.scale(loss).backward()

                    # if ((batch_idx + 1) % self.accumulation_iters == 0) or ((batch_idx + 1) == len(self.train_loader)):
                    if (batch_idx + 1) % self.accumulation_iters == 0:
                        self.grad_scaler.step(self.optimizer)
                        self.grad_scaler.update()
                        self.optimizer.zero_grad(set_to_none=True)
                        self.ema.ema_step(ema_model=self.ema_model, model=self.unet_model)

                        # if epoch > self.swa_start:
                        #     self.swa_model.update_parameters(model=self.unet_model)
                        #     self.swa_scheduler.step()
                        # else:
                        #     self.scheduler.step()

                        # total_loss += (float(accumulated_minibatch_loss) / len(self.train_loader) * self.accumulation_iters)
                        pbar.set_description(
                            # f'Loss minibatch: {float(accumulated_minibatch_loss):.4f}, total: {total_loss:.4f}'
                            f'Loss minibatch: {float(accumulated_minibatch_loss):.4f}'
                        )
                        accumulated_minibatch_loss = 0.0

                    if not batch_idx % self.save_every:
                        self.sample(epoch=epoch, batch_idx=batch_idx, sample_count=self.sample_count)

                        Utils.save_checkpoint(
                            epoch=epoch,
                            model=self.unet_model,
                            optimizer=self.optimizer,
                            scheduler=self.scheduler,
                            grad_scaler=self.grad_scaler,
                            filename=os.path.join(self.save_path, f'model_{epoch}_{batch_idx}.pt')
                        )
                        Utils.save_checkpoint(
                            epoch=epoch,
                            model=self.ema_model,
                            filename=os.path.join(self.save_path, f'model_ema_{epoch}_{batch_idx}.pt')
                        )

            self.scheduler.step()


if __name__ == '__main__':
    trainer = Trainer(
        dataset_path=r'C:\datasets\cars',
        save_path=r'C:\DeepLearningPytorch\ddpm',
        # checkpoint_path=r'C:\DeepLearningPytorch\ddpm\model_126_0.pt',
        # checkpoint_path_ema=r'C:\DeepLearningPytorch\ddpm\model_ema_126_0.pt',
    )
    trainer.train()

    # trainer.sample(output_name='output6', sample_count=4)

    # trainer.sample_gif(
    #     output_name='output8',
    #     sample_count=1,
    #     save_path=r'C:\DeepLearningPytorch\ddpm'
    # )

    # tester = Tester(device='cuda')
    # tester.test_unet()
    # tester.test_attention()
    # tester.test_jit()
