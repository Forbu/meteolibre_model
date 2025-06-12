"""
In this module we will test a new architecture for the VAE.
IMG -> Encoder (CNN + DiT) -> Latent Space -> Decoder (DiT + CNN) -> IMG
"""

import os
import glob

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl

import einops

from torch.optim import optimizer

import wandb
from PIL import Image

from heavyball import ForeachSOAP

from diffusers import AutoencoderKL
from timm.models.vision_transformer import PatchEmbed

from dit_ml.dit import DiT


class VAEMeteoLibrePLModelDitVae(pl.LightningModule):
    """
    PyTorch Lightning module for the MeteoLibre model.

    VAE autoencoder
    """

    def __init__(
        self,
        learning_rate=1e-6,
        test_dataloader=None,
        dir_save="../",
        input_channels=5,
        output_channels=5,
        latent_dim=64,
        coefficient_reg=0.01,
    ):
        """
        Initialize the MeteoLibrePLModel.

        Args:
            TODO later
        """
        super().__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.latent_dim = latent_dim
        self.coefficient_reg = coefficient_reg
        self.learning_rate = learning_rate
        self.patch_size = 2

        self.model = AutoencoderKL(
            in_channels=input_channels,
            out_channels=output_channels,
            latent_channels=latent_dim,
            act_fn="silu",
            block_out_channels=[128 // 4, 256 // 4, 512 // 4, 512 // 4],
            down_block_types=[
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
            ],
            layers_per_block=2,
            sample_size=256,
            scaling_factor=0.18215,
            up_block_types=[
                "UpDecoderBlock2D",
                "UpDecoderBlock2D",
                "UpDecoderBlock2D",
                "UpDecoderBlock2D",
            ],
        )

        self.patch_embedder = PatchEmbed(
            img_size=32, patch_size=2, in_chans=latent_dim, embed_dim=128
        )

        self.dit_encoder = DiT(
            num_patches=32 * 32 // 4,  # if 2d with flatten size
            hidden_size=128,
            depth=3,
            num_heads=8,
            causal_block=True,
            causal_block_size=32 * 32 // 4,
        )

        self.dit_decoder = DiT(
            num_patches=32 * 32 // 4,  # if 2d with flatten size
            hidden_size=128,
            depth=3,
            num_heads=8,
        )

        self.final_layer = nn.Linear(128, 2 * 2 * latent_dim, bias=True)

        self.learning_rate = learning_rate
        self.test_dataloader = test_dataloader

        self.dir_save = dir_save

    def forward(self, x_image, mask_values):
        """
        Forward pass through the model.

        Args:
            x_image (torch.Tensor): Input image tensor of size (b, nb_frame, c, h, w)

        Returns:
            torch.Tensor: Output tensor from the model.
        """
        batch_size, nb_frame, c, h, w = x_image.shape

        # flatten to get (batch_size * nb_frame, c, h, w)
        x_image = einops.rearrange(x_image, "b t c h w -> (b t) c h w")
        latents_sample = self.model.encode(x_image).latent_dist.mean

        # 1. patchify
        latents_sample_patch = self.patch_embedder(
            latents_sample
        )  # size is (batch_size * nb_frame, H * W, embed_dim)



        # 2. pass though DiT
        # resize to (batch_size, H * W * nb_frame, embed_dim)
        latents_sample_patch = einops.rearrange(
            latents_sample_patch, "(b t) nb_patch c -> b (t nb_patch) c", t=nb_frame
        )

        dummy_time = torch.zeros(
            (latents_sample_patch.shape[0], 128),
            device=latents_sample_patch.device,
            dtype=torch.float32,
        )

        # pass through DiT encoder
        dit_encoded_latent = self.dit_encoder(latents_sample_patch, dummy_time)

        # pass through DiT decoder
        dit_decoded_latent = self.dit_decoder(dit_encoded_latent, dummy_time)

        decoder_latents = einops.rearrange(
            dit_decoded_latent, "b (t nb_patch) c -> (b t) nb_patch c", t=nb_frame
        )

        # 3. unpatchify
        decoder_latents_final = self.final_layer(
            decoder_latents
        )  # (N, T, patch_size ** 2 * out_channels)

        decoder_latents_unpatch = self.unpatchify(decoder_latents_final)

        # decoder cnn anti
        final_image = self.model.decode(decoder_latents_unpatch).sample

        # reshape
        final_image = einops.rearrange(
            final_image, "(b t) c h w -> b t c h w", t=nb_frame
        )

        return final_image, (dit_encoded_latent)

    def training_step(self, batch, batch_idx):
        """
        Training step for the PyTorch Lightning module.
        """

        radar_data = batch["radar_back"].unsqueeze(-1)
        groundstation_data = batch["groundstation_back"]

        # little correction
        groundstation_data = torch.where(
            groundstation_data == -100, -1, groundstation_data
        )

        # mask radar
        mask_radar = torch.ones_like(radar_data)
        mask_groundstation = groundstation_data != -100

        # concat the two elements
        x_image = torch.cat((radar_data, groundstation_data), dim=-1)
        x_mask = torch.cat((mask_radar, mask_groundstation), dim=-1)

        x_image = x_image.permute(0, 1, 4, 2, 3)  # (N, nb_frame, C, H, W)
        x_mask = x_mask.permute(0, 1, 4, 2, 3)  # (N, nb_frame, C, H, W))

        # forward pass
        final_image, latent = self(x_image, x_mask)

        reconstruction_loss = F.mse_loss(final_image, x_image, reduction="none")
        reconstruction_loss = reconstruction_loss * x_mask
        reconstruction_loss = reconstruction_loss.sum() / x_mask.sum()

        self.log("reconstruction_loss", reconstruction_loss)



        regularization_loss = F.mse_loss(
            latent, torch.zeros_like(latent), reduction="mean"
        )

        self.log("regularization_loss", regularization_loss)

        print("reconstruction_loss:", reconstruction_loss.item())
        print("regularization_loss:", regularization_loss.item())

        loss = reconstruction_loss + self.coefficient_reg * regularization_loss

        return loss

    def configure_optimizers(self):
        """
        Configures the optimizer for training.

        Returns:
            torch.optim.Optimizer: Adam optimizer.
        """
        # optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        optimizer = ForeachSOAP(self.parameters(), lr=self.learning_rate, foreach=False)
        return optimizer

    @torch.no_grad()
    def generate_one(self, nb_batch=1, nb_step=100):
        # Assuming batch is a dictionary returned by TFDataset
        # generate a random (nb_batch, 1, 256, 256) tensor
        for batch in self.test_dataloader:
            break

        
        radar_data = batch["radar_back"].unsqueeze(-1)
        groundstation_data = batch["groundstation_back"]

        # little correction
        groundstation_data = torch.where(
            groundstation_data == -100, -1, groundstation_data
        )

        # mask radar
        mask_radar = torch.ones_like(radar_data)
        mask_groundstation = groundstation_data != -100

        # concat the two elements
        x_image = torch.cat((radar_data, groundstation_data), dim=-1)
        x_mask = torch.cat((mask_radar, mask_groundstation), dim=-1)

        x_image = x_image.permute(0, 1, 4, 2, 3)  # (N, nb_frame, C, H, W)
        x_mask = x_mask.permute(0, 1, 4, 2, 3)  # (N, nb_frame, C, H, W))

        # forward pass
        final_image, latent = self(x_image, x_mask)

        return final_image[:, 0, :, :, :].permute(0, 2, 3, 1), x_image[
            :, 0, :, :, :
        ].permute(0, 2, 3, 1)

    # on epoch end of training
    def on_train_epoch_end(self):
        # generate image
        self.eval()

        result, x_image_future = self.generate_one(nb_batch=1, nb_step=100)

        self.save_image(result[0, :, :, 0], name_append="result")
        self.save_image(x_image_future[0, :, :, 0], name_append="target")

        self.save_gif(result, name_append="result_gif_vae")
        self.save_gif(x_image_future, name_append="target_gif_vae")

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

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.latent_dim
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs
