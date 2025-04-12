"""
meteolibre_model/meteolibre_model/pl_model.py
"""

import os
import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl

import einops

import matplotlib.pyplot as plt

from torch.optim import optimizer
import wandb


from heavyball import ForeachSOAP
from diffusers import AutoencoderKLAllegro


class VAEMeteoLibrePLModelGrid(pl.LightningModule):
    """
    PyTorch Lightning module for the MeteoLibre model.

    VAE autoencoder
    """

    def __init__(
        self,
        learning_rate=1e-3,
        beta=0.000005,
        test_dataloader=None,
        dir_save="../",
    ):
        """
        Initialize the MeteoLibrePLModel.

        Args:
            input_channels_ground (int): Number of input channels for the ground station image.
            condition_size (int): Size of the conditioning vector.
            learning_rate (float, optional): Learning rate for the optimizer. Defaults to 1e-3.
            nb_back (int, optional): Number of past frames to use as input. Defaults to 3.
            nb_future (int, optional): Number of future frames to predict. Defaults to 1.
        """
        super().__init__()
        self.model = AutoencoderKLAllegro(
            in_channels=1,
            out_channels=1,
            latent_channels=4,
            temporal_compression_ratio=1,
            block_out_channels = (128//2, 256//2, 512//2, 512//2)
        ).float()

        self.model.enable_slicing()
        self.model.enable_tiling()

        self.learning_rate = learning_rate
        self.test_dataloader = test_dataloader
        self.beta = beta

        self.dir_save = dir_save

    def forward(self, x_image):
        """
        Forward pass through the model.

        Args:
            x_image (torch.Tensor): Input image tensor.

        Returns:
            torch.Tensor: Output tensor from the model.
        """
        latents_sample = self.model.encode(x_image).latent_dist
        latents_mean = latents_sample.mean
        latents_variance = latents_sample.logvar

        final_image = self.model.decode(latents_sample.sample()).sample

        return final_image, (latents_mean, latents_variance)

    def training_step(self, batch, batch_idx):
        """
        Training step for the PyTorch Lightning module.

        """
        x_image = batch["target_radar_frames"][:, :, :, :]

        # rearrange to (batch_size, 2, 256, 256)
        # and then to (batch_size * 2, 1, 256, 256)
        x_image = einops.rearrange(x_image, "b h w t -> b t h w")
        x_image = x_image.unsqueeze(1).float()  # size is b c t h w

        # Forward pass through the model
        final_image, (latents_mean, latents_variance) = self(x_image)

        reconstruction_loss = F.mse_loss(final_image, x_image)

        self.log("reconstruction_loss", reconstruction_loss)

        kl_loss = torch.mean(
            -0.5
            * torch.sum(
                1 + latents_variance - latents_mean**2 - latents_variance.exp(), dim=1
            ),
        )

        self.log("kl_loss", kl_loss)

        loss = reconstruction_loss + self.beta * kl_loss

        return loss

    def configure_optimizers(self):
        """
        Configures the optimizer for training.

        Returns:
            torch.optim.Optimizer: Adam optimizer.
        """
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        # optimizer = ForeachSOAP(self.parameters(), lr=self.learning_rate, foreach=False)
        return optimizer

    @torch.no_grad()
    def generate_one(self, nb_batch=1, nb_step=100):
        # Assuming batch is a dictionary returned by TFDataset
        # generate a random (nb_batch, 1, 256, 256) tensor
        for batch in self.test_dataloader:
            break

        x_image = batch["target_radar_frames"][:, :, :, :]
        x_image = x_image.permute(0, 3, 1, 2)
        x_image = x_image.unsqueeze(1).to(self.device)

        # Forward pass through the model
        final_image, _ = self(x_image.float())

        return final_image[:, :, 0, :, :].permute(0, 2, 3, 1), x_image[
            :, :, 0, :, :
        ].permute(0, 2, 3, 1)

    # on epoch end of training
    def on_train_epoch_end(self):
        # generate image
        self.eval()

        result, x_image_future = self.generate_one(nb_batch=1, nb_step=100)

        self.save_image(result[0, :, :, 0].cpu().numpy(), name_append="result")
        self.save_image(x_image_future[0, :, :, 0].cpu().numpy(), name_append="future")

        self.train()

    def save_image(self, result, name_append="result"):
        radar_image = result

        fname = (
            self.dir_save + f"data/{name_append}_radar_epoch_{self.current_epoch}.png"
        )

        plt.figure(figsize=(20, 20))
        plt.imshow(radar_image, vmin=-1, vmax=2)
        plt.colorbar()

        plt.savefig(fname, bbox_inches="tight", pad_inches=0)
        plt.close()

        # logging image into wandb
        self.logger.log_image(
            key=name_append, images=[wandb.Image(fname)], caption=[name_append]
        )
