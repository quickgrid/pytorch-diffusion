"""Implementation of DDIM.

References:
    - Annotated DDPM implementation,
        https://github.com/quickgrid/paper-implementations/tree/main/pytorch/denoising-diffusion.
    - Keras DDIM,
        https://keras.io/examples/generative/ddim/.
"""
import copy
import math
import os
import logging
import pathlib
from typing import Tuple, Union, List

import torch
import torch.nn as nn
import torchvision.utils
from PIL import Image
from torch.cuda.amp import GradScaler
from torch.utils.checkpoint import checkpoint
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
            noise_steps: int,
            min_signal_rate: int = 0.02,
            max_signal_rate: int = 0.95,
    ):
        self.max_signal_rate = max_signal_rate
        self.min_signal_rate = min_signal_rate
        self.device = device
        self.noise_steps = noise_steps
        self.img_size = img_size

    def diffusion_schedule(
            self,
            diffusion_times,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        max_signal_rate = torch.tensor(self.max_signal_rate, dtype=torch.float, device=self.device)
        min_signal_rate = torch.tensor(self.min_signal_rate, dtype=torch.float, device=self.device)

        start_angle = torch.acos(max_signal_rate)
        end_angle = torch.acos(min_signal_rate)

        diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)

        signal_rates = torch.cos(diffusion_angles)
        noise_rates = torch.sin(diffusion_angles)

        return noise_rates, signal_rates

    @staticmethod
    def denoise(
            eps_model: nn.Module,
            noisy_images: torch.Tensor,
            noise_rates: torch.Tensor,
            signal_rates: torch.Tensor,
            training: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict noise component and calculate the image component using it.
        """
        if training:
            pred_noises = eps_model(noisy_images, noise_rates.to(dtype=torch.long) ** 2)
            pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates
            return pred_noises, pred_images

        with torch.no_grad():
            pred_noises = eps_model(noisy_images, noise_rates.to(dtype=torch.long) ** 2)
            pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates

        return pred_noises, pred_images

    def reverse_diffusion(
            self,
            num_images: int,
            diffusion_steps: int,
            eps_model: nn.Module,
            scale_factor: int = 2,
            sample_gif: bool = False,
    ) -> Union[torch.Tensor, List[Image.Image]]:
        eps_model.eval()

        frames_list = []
        pred_images = None
        initial_noise = torch.randn((num_images, 3, self.img_size, self.img_size), device=self.device)

        step_size = 1.0 / diffusion_steps

        next_noisy_images = initial_noise
        for step in range(diffusion_steps):
            noisy_images = next_noisy_images

            diffusion_times = torch.ones((num_images, 1, 1, 1), device=self.device) - step * step_size
            noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
            pred_noises, pred_images = self.denoise(
                eps_model, noisy_images, noise_rates, signal_rates, training=False
            )

            if sample_gif:
                output = ((pred_images.clamp(-1, 1) + 1) * 127.5).type(torch.uint8)
                output = F.interpolate(input=output, scale_factor=scale_factor, mode='nearest-exact')
                grid = torchvision.utils.make_grid(output)
                img_arr = grid.permute(1, 2, 0).cpu().numpy()
                img = Image.fromarray(img_arr)
                frames_list.append(img)

            next_diffusion_times = diffusion_times - step_size
            next_noise_rates, next_signal_rates = self.diffusion_schedule(next_diffusion_times)
            next_noisy_images = (next_signal_rates * pred_images + next_noise_rates * pred_noises)

        eps_model.train()

        if sample_gif:
            return frames_list

        pred_images = ((pred_images.clamp(-1, 1) + 1) * 127.5).type(torch.uint8)
        pred_images = F.interpolate(input=pred_images, scale_factor=scale_factor, mode='nearest-exact')
        return pred_images


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

        self.out_channels = out_channels

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(in_features=emb_dim, out_features=out_channels),
        )

    def forward(self, x: torch.Tensor, t_embedding: torch.Tensor) -> torch.Tensor:
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t_embedding)
        emb = emb.permute(0, 3, 1, 2).repeat(1, 1, x.shape[-2], x.shape[-1])
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
        emb = emb.permute(0, 3, 1, 2).repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int = None, dropout: float = 0.):
        super(MLP, self).__init__()
        hidden_dim = hidden_dim or dim
        self.net = nn.Sequential(
            nn.Linear(in_features=dim, out_features=hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden_dim, out_features=dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerEncoderSA(nn.Module):
    def __init__(self, num_channels: int, size: int, num_heads: int = 4, hidden_dim: int = 1024, dropout: int = 0.0):
        """A block of transformer encoder with mutli head self attention from vision transformers paper,
         https://arxiv.org/pdf/2010.11929.pdf.
        """
        super(TransformerEncoderSA, self).__init__()
        self.num_channels = num_channels
        self.size = size
        self.mha = nn.MultiheadAttention(embed_dim=num_channels, num_heads=num_heads, batch_first=True)
        self.ln_1 = nn.LayerNorm([num_channels])
        self.ln_2 = nn.LayerNorm([num_channels])
        self.mlp = MLP(dim=num_channels, hidden_dim=hidden_dim, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, self.num_channels, self.size * self.size).permute(0, 2, 1)
        x_ln = self.ln_1(x)
        attention_value, _ = self.mha(query=x_ln, key=x_ln, value=x_ln)
        x = attention_value + x
        x = self.mlp(self.ln_2(x)) + x
        return x.permute(0, 2, 1).view(-1, self.num_channels, self.size, self.size)


class UNet(nn.Module):
    def __init__(
            self,
            noise_steps: int,
            in_channels: int = 3,
            out_channels: int = 3,
            time_dim: int = 256,
    ):
        super(UNet, self).__init__()
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
            t: Time step defined as long integer.
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

        # x = checkpoint(self.sa6, x)
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
    def save_images(images: torch.Tensor, save_path: str, nrow: int = 8) -> None:
        grid = torchvision.utils.make_grid(images, nrow=nrow)
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
        checkpoint_dict = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
        }
        if optimizer:
            checkpoint_dict['optimizer'] = optimizer.state_dict()
        if scheduler:
            checkpoint_dict['scheduler'] = scheduler.state_dict()
        if scheduler:
            checkpoint_dict['grad_scaler'] = grad_scaler.state_dict()

        torch.save(checkpoint_dict, filename)
        logging.info("=> Saving checkpoint complete.")

    @staticmethod
    def load_checkpoint(
            model: nn.Module,
            filename: str,
            enable_train_mode: bool,
            optimizer: optim.Optimizer = None,
            scheduler: optim.lr_scheduler = None,
            grad_scaler: GradScaler = None,
    ) -> int:
        logging.info("=> Loading checkpoint")
        saved_model = torch.load(filename, map_location="cuda")
        model.load_state_dict(saved_model['state_dict'], strict=False)
        if 'optimizer' in saved_model and enable_train_mode:
            optimizer.load_state_dict(saved_model['optimizer'])
        if 'scheduler' in saved_model and enable_train_mode:
            scheduler.load_state_dict(saved_model['scheduler'])
        if 'grad_scaler' in saved_model and enable_train_mode:
            grad_scaler.load_state_dict(saved_model['grad_scaler'])
        return saved_model['epoch']


class DDIM:
    def __init__(
            self,
            dataset_path: str,
            save_path: str = None,
            checkpoint_path: str = None,
            checkpoint_path_ema: str = None,
            run_name: str = 'ddpm',
            image_size: int = 64,
            image_channels: int = 3,
            accumulation_batch_size: int = 2,
            accumulation_iters: int = 16,
            sample_count: int = 1,
            num_workers: int = 0,
            device: str = 'cuda',
            num_epochs: int = 10000,
            fp16: bool = False,
            save_every: int = 500,
            learning_rate: float = 2e-4,
            noise_steps: int = 500,
            enable_train_mode: bool = True,
    ):
        self.num_epochs = num_epochs
        self.device = device
        self.fp16 = fp16
        self.save_every = save_every
        self.accumulation_iters = accumulation_iters
        self.sample_count = sample_count
        self.accumulation_batch_size = accumulation_batch_size
        self.enable_train_mode = enable_train_mode

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
            batch_size=accumulation_batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=num_workers,
            drop_last=False,
            collate_fn=Utils.collate_fn,
        )

        self.unet_model = UNet(noise_steps=noise_steps).to(device)
        self.diffusion = Diffusion(img_size=image_size, device=self.device, noise_steps=noise_steps)
        self.optimizer = optim.Adam(
            params=self.unet_model.parameters(), lr=learning_rate,  # betas=(0.9, 0.999)
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer, T_max=300)
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
                enable_train_mode=enable_train_mode,
            )
        if checkpoint_path_ema:
            logging.info(f'Loading EMA model weights...')
            _ = Utils.load_checkpoint(
                model=self.ema_model,
                filename=checkpoint_path_ema,
                enable_train_mode=enable_train_mode,
            )

    def sample(
            self,
            epoch: int = None,
            batch_idx: int = None,
            sample_count: int = 1,
            output_name: str = None,
            diffusion_steps: int = 40,
    ) -> None:
        """Generates images with reverse process based on sampling method with both training model and ema model.
        """
        sampled_images = self.diffusion.reverse_diffusion(
            eps_model=self.unet_model, num_images=sample_count, diffusion_steps=diffusion_steps,
        )
        ema_sampled_images = self.diffusion.reverse_diffusion(
            eps_model=self.ema_model, num_images=sample_count, diffusion_steps=diffusion_steps,
        )

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
            output_name: str,
            save_path: str = '',
            sample_count: int = 1,
            diffusion_steps: int = 40,
            optimize: bool = False,
    ) -> None:
        """Generates images with reverse process based on sampling method with both training model and ema model.
        """
        sampled_images = self.diffusion.reverse_diffusion(
            eps_model=self.unet_model, num_images=sample_count, sample_gif=True, diffusion_steps=diffusion_steps,
        )
        ema_sampled_images = self.diffusion.reverse_diffusion(
            eps_model=self.ema_model, num_images=sample_count, sample_gif=True, diffusion_steps=diffusion_steps,
        )

        model_name = f'{output_name}.gif'
        sampled_images[0].save(
            os.path.join(save_path, model_name),
            save_all=True,
            append_images=sampled_images[1:],
            optimize=optimize,
            duration=80,
            loop=0
        )

        ema_model_name = f'{output_name}_ema.gif'
        ema_sampled_images[0].save(
            os.path.join(save_path, ema_model_name),
            save_all=True,
            append_images=ema_sampled_images[1:],
            optimize=optimize,
            duration=80,
            loop=0
        )

    def train(self) -> None:
        assert self.enable_train_mode, 'Cannot train when enable_train_mode flag disabled.'

        logging.info(f'Training started....')
        for epoch in range(self.start_epoch, self.num_epochs):
            accumulated_minibatch_loss = 0.0
            accumulated_image_loss = 0.0
            # accumulated_image_ema_loss = 0.0

            with tqdm(self.train_loader) as pbar:
                for batch_idx, (real_images, _) in enumerate(pbar):
                    real_images = real_images.to(self.device)
                    current_batch_size = real_images.shape[0]

                    noises = torch.randn(size=(current_batch_size, 3, 64, 64), device=self.device)

                    # sample uniform random diffusion times
                    diffusion_times = torch.rand(size=(current_batch_size, 1, 1, 1), device=self.device)

                    noise_rates, signal_rates = self.diffusion.diffusion_schedule(diffusion_times)
                    # mix the images with noises accordingly
                    noisy_images = signal_rates * real_images + noise_rates * noises

                    with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=self.fp16):

                        pred_noises, pred_images = self.diffusion.denoise(
                            self.unet_model, noisy_images, noise_rates, signal_rates, training=True
                        )

                        # pred_noises_ema, pred_images_ema = self.diffusion.denoise(
                        #     self.ema_model, noisy_images, noise_rates, signal_rates, training=True
                        # )

                        loss = F.smooth_l1_loss(pred_noises, noises)
                        loss /= self.accumulation_iters

                        accumulated_minibatch_loss += float(loss)
                        accumulated_image_loss += (F.smooth_l1_loss(pred_images, real_images) / self.accumulation_iters)
                        # accumulated_image_ema_loss += (F.smooth_l1_loss(pred_images_ema, real_images) / self.accumulation_iters)

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

                        pbar.set_description(
                            f'Loss => '
                            f'Noise: {float(accumulated_minibatch_loss):.4f}, '
                            f'Image: {accumulated_image_loss:.4f} '
                            # f'Image EMA: {accumulated_image_ema_loss:.4f} '
                        )
                        accumulated_minibatch_loss = 0.0
                        accumulated_image_loss = 0.0
                        # accumulated_image_ema_loss = 0.0

                    if not batch_idx % self.save_every:
                        real_images_out = ((real_images.clamp(-1, 1) + 1) * 127.5).type(torch.uint8)
                        noisy_images_out = ((noisy_images.clamp(-1, 1) + 1) * 127.5).type(torch.uint8)
                        pred_images_out = ((pred_images.clamp(-1, 1) + 1) * 127.5).type(torch.uint8)
                        images_out = torch.cat([real_images_out, noisy_images_out, pred_images_out], dim=0)
                        images_out = F.interpolate(input=images_out, scale_factor=2, mode='nearest-exact')

                        Utils.save_images(
                            images=images_out,
                            save_path=os.path.join(self.save_path, 'real_noised_denoised.jpg'),
                            nrow=self.accumulation_batch_size,
                        )

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
    ddim = DDIM(
        dataset_path=r'C:\computer_vision\celeba',
        save_path=r'C:\computer_vision\ddim',
        checkpoint_path=r'C:\computer_vision\ddim\ddim_celeba_66_0.pt',
        checkpoint_path_ema=r'C:\computer_vision\ddim\ddim_celeba_ema_66_0.pt',
        # enable_train_mode=False,
    )
    ddim.train()

    # ddim.sample(output_name='output9', sample_count=2, diffusion_steps=40)

    # ddim.sample_gif(
    #     output_name='output8',
    #     sample_count=1,
    #     save_path=r'C:\computer_vision\ddim',
    #     diffusion_steps=40,
    # )
