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

from PIL import Image


from heavyball import ForeachSOAP
from diffusers import AutoencoderKLHunyuanVideo


class VAEMeteoLibrePLModelGrid(pl.LightningModule):
    """
    PyTorch Lightning module for the MeteoLibre model.

    VAE autoencoder
    """

    def __init__(
        self,
        learning_rate=1e-3,
        beta=0.000001,
        test_dataloader=None,
        dir_save="./",
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
        self.model = AutoencoderKLHunyuanVideo(
            in_channels=1,
            out_channels=1,
            latent_channels=4,
            temporal_compression_ratio= 4,
            block_out_channels = (128//4, 256//4, 512//4, 512//4)
        ).float()

        self.model.train()

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
        #optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        optimizer = ForeachSOAP(self.parameters(), lr=self.learning_rate, foreach=False)
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

        return final_image[:, 0, :, :, :].permute(0, 2, 3, 1), x_image[
            :, 0, :, :, :
        ].permute(0, 2, 3, 1)

    # on epoch end of training
    def on_train_epoch_end(self):
        # generate image
        self.eval()

        with torch.no_grad():

            result, x_image_future = self.generate_one(nb_batch=1, nb_step=100)
            # reshape first
            self.save_image(result[0, :, :, 0], name_append="result")
            self.save_image(x_image_future[0, :, :, 0], name_append="target")

            self.save_gif(result, name_append="result_gif_vae")
            self.save_gif(x_image_future, name_append="target_gif_vae")

            # now we delete all the png files (not the gif)
            for f in glob.glob(self.dir_save + "data/*.png"):
                os.remove(f)


    def save_gif(self, result, name_append="result", duration=10):
        nb_frame = result.shape[-1]

        file_name_list = []

        for i in range(nb_frame):
            fname = (
                self.dir_save
                + f"data/{name_append}_radar_epoch_{self.current_epoch}_{i}.png"
            )

            radar_image = result[0, :, :, i].cpu().numpy()


            plt.figure(figsize=(20, 20))
            plt.imshow(radar_image, vmin=-1, vmax=2)
            plt.colorbar()

            plt.savefig(fname, bbox_inches="tight", pad_inches=0)
            plt.close()

            file_name_list.append(fname)

        create_gif_pillow(
            image_paths=file_name_list,
            output_path=self.dir_save
            + f"data/{name_append}_radar_epoch_{self.current_epoch}.gif",
            duration=duration,
        )

    def save_image(self, result, name_append="result"):
        radar_image = result.cpu().numpy()

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



def create_gif_pillow(image_paths, output_path, duration=100):
    """
    Creates a GIF from a list of image paths using Pillow.

    Args:
      image_paths: A list of strings, where each string is the path to an image file.
      output_path: The path where the GIF will be saved (e.g., 'output.gif').
      duration: The duration (in milliseconds) to display each frame in the GIF.
    """
    images = []
    for path in image_paths:
        try:
            img = Image.open(path)
            images.append(img)
        except FileNotFoundError:
            print(f"Error: Image not found at {path}")
            return

    if images:
        first_frame = images[0]
        remaining_frames = images[1:]

        first_frame.save(
            output_path,
            save_all=True,
            append_images=remaining_frames,
            duration=duration,
            loop=0,  # 0 means loop indefinitely
        )
        print(f"GIF created successfully at {output_path}")
    else:
        print("No valid images found to create GIF.")
