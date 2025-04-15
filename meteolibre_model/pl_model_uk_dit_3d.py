"""
meteolibre_model/meteolibre_model/pl_model.py
"""

import os
import glob
import wandb
import matplotlib.pyplot as plt
from PIL import Image

from mpmath.libmp import prec_to_dps

import torch
import torch.nn as nn
import einops
from torch.optim import optimizer

import lightning.pytorch as pl
from diffusers import AutoencoderKL, AutoencoderKLHunyuanVideo


from meteolibre_model.model_3Dtransformer import TransfomerFilmModel

from heavyball import ForeachSOAP


NORMALIZATION_FACTOR = 0.769


class MeteoLibrePLModelGrid(pl.LightningModule):
    """
    PyTorch Lightning module for the MeteoLibre model.

    This class wraps the SimpleConvFilmModel and integrates it with PyTorch Lightning
    for training and evaluation. It handles data loading, optimization, and defines
    the training step using a Rectified Flow approach with MSE loss.
    """

    def __init__(
        self,
        condition_size,
        learning_rate=1e-4,
        nb_back=5,
        nb_future=4,
        shape_image=512,
        test_dataloader=None,
        dir_save="../",
        loss_type="mse",
        parametrization="endpoint",
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
        self.model = TransfomerFilmModel(
            4,
            4,
            condition_size,
        )
        self.learning_rate = learning_rate
        self.criterion = nn.MSELoss(reduction="none")  # Rectified Flow uses MSE loss

        self.test_dataloader = test_dataloader

        self.dir_save = dir_save

        self.nb_back = nb_back
        self.nb_future = nb_future

        self.nb_back_vae = ((self.nb_back - 1) // 4 + 1)

        self.shape_image = shape_image
        self.loss_type = loss_type
        self.parametrization = parametrization

        self.fn_loss = nn.MSELoss(reduction="none")

        #self.vae = AutoencoderKL.from_pretrained("Forbu14/meteolibre", subfolder="weights_vae")
        self.vae = AutoencoderKLHunyuanVideo.from_pretrained("Forbu14/meteolibre", subfolder="weights_vae_3d")

    def forward(self, x_image, x_scalar):
        """
        Forward pass through the model.

        Args:
            x_image (torch.Tensor): Input image tensor.
            x_scalar (torch.Tensor): Input scalar condition tensor.

        Returns:
            torch.Tensor: Output tensor from the model.
        """
        return self.model(x_image, x_scalar)

    def getting_target_input_after_vae(self, batch):
        with torch.no_grad():
            target_radar_frames = batch[
                "target_radar_frames"
            ]  # (batch_size, 256, 256, nb_future)
            target_radar_frames = target_radar_frames.unsqueeze(1) # (b c h w t)
            target_radar_frames = einops.rearrange(
                target_radar_frames, "b c h w t -> b c t h w"
            )

            input_radar_frames = batch[
                "input_radar_frames"
            ]  # (batch_size, 256, 256, nb_back)
            input_radar_frames = input_radar_frames.unsqueeze(1)
            input_radar_frames = einops.rearrange(
                input_radar_frames, "b c h w t -> b c t h w"
            )

            full_frame = torch.cat( 
                [input_radar_frames, target_radar_frames], dim=2
            )

            full_frame = self.vae.encode(
                full_frame
            ).latent_dist.sample()

            full_frame = full_frame * NORMALIZATION_FACTOR

            target_radar_frames = full_frame[:, :, self.nb_back_vae :, :, :]
            input_radar_frames = full_frame[:, :, : self.nb_back_vae, :, :]

            # detach to avoid backpropagation
            target_radar_frames = target_radar_frames.detach()
            input_radar_frames = input_radar_frames.detach()

        return target_radar_frames, input_radar_frames

    def init_prior(self, batch_size, shape_target):
        return torch.randn((shape_target)).to(self.device)

    def training_step(self, batch, batch_idx):
        """
        Training step for the PyTorch Lightning module.

        This method defines the training logic for a single batch. It involves:
        1. Preparing input tensors from the batch.
        2. Projecting and embedding the ground station image.
        3. Concatenating historical images.
        4. Sampling prior noise and time variable for Rectified Flow.
        5. Interpolating between prior and future image to get the diffused sample x_t.
        6. Predicting the velocity field v_t.
        7. Calculating the target velocity field.
        8. Computing MSE loss between predicted and target velocity fields.
        9. Masking the loss for ground station data.
        10. Logging training loss.

        Args:
            batch (dict): A dictionary containing the training batch data,
                          expected to be returned by TFDataset.
            batch_idx (int): Index of the current batch.

        Returns:
            torch.Tensor: Training loss for the batch.
        """
        # Assuming batch is a dictionary returned by TFDataset
        # and contains 'back_0', 'future_0', 'hour' keys

        target_radar_frames, input_radar_frames = self.getting_target_input_after_vae(
            batch
        )

        # Prior sample (simple Gaussian noise) - you can refine this prior
        prior_image = self.init_prior(
            target_radar_frames.shape[0], target_radar_frames.shape
        )

        # Time variable for Rectified Flow - sample uniformly
        t = (
            torch.rand(target_radar_frames.shape[0], 1, 1, 1, 1)
            .type_as(target_radar_frames)
            .to(target_radar_frames.device)
        )  # (B, 1)

        # we create a scalar value to condition the model on time stamp
        # and hours
        x_hour = batch["hour"].clone().detach().float().unsqueeze(1)  # (B, 1)
        x_minute = batch["minute"].clone().detach().float().unsqueeze(1)  # (B, 1)

        # Simple scalar condition: hour of the day and minutes of the day. You might want to expand this.
        x_scalar = torch.cat([x_hour, x_minute, t[:, :, 0, 0, 0]], dim=1)

        # Interpolate between prior and data to get x_t
        x_t = t * target_radar_frames + (1 - t) * prior_image

        # concat x_t with x_image_back and x_ground_station_image_previous
        input_model = torch.cat([input_radar_frames, x_t], dim=2)

        if self.parametrization == "endpoint":
            # Predict endpoint field v_t using the model
            pred = self.forward(input_model, x_scalar)

            # coefficient of ponderation for endpoint parametrization
            w_t = t / (1 - t)
            w_t = torch.clamp(w_t, min=0.05, max=2.0)

            target = target_radar_frames

        elif self.parametrization == "noisy":
            # prediction the noise
            pred = self.forward(input_model, x_scalar)

            # coefficient of ponderation for noisy parametrization
            w_t = (1 / (t + 0.0001)) ** 2

            w_t = torch.clamp(w_t, min=1.0, max=3.0)

            target = prior_image

        # Loss is MSE between predicted and target velocity fields
        loss = self.fn_loss(pred[:, :, self.nb_back_vae :, :, :], target)
        loss = w_t * loss  # ponderate loss

        loss = loss.mean()

        # Log the lossruff
        self.log("train_loss", loss)

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
        # we first generate a random noise
        for batch in self.test_dataloader:
            break

        # convert to device
        batch = {k: v.to(self.device) for k, v in batch.items()}

        batch_size = batch["target_radar_frames"].shape[0]

        target_radar_frames, input_radar_frames = self.getting_target_input_after_vae(
            batch
        )

        tmp_noise = self.init_prior(
            target_radar_frames.shape[0], target_radar_frames.shape
        )

        for i in range(1, nb_step):
            # concat x_t with x_image_back and x_ground_station_image_previous
            input_model = torch.cat([input_radar_frames, tmp_noise], dim=1)

            t = i * 1.0 / nb_step

            x_scalar = torch.cat(
                [
                    batch["hour"].clone().detach().float().unsqueeze(1),
                    batch["minute"].clone().detach().float().unsqueeze(1),
                    torch.ones(batch_size, 1)
                    .type_as(batch["target_radar_frames"])
                    .to(self.device)
                    * t,
                ],
                dim=1,
            )

            if self.parametrization == "endpoint":
                endpoint = self.forward(input_model, x_scalar)

                velocity = (
                    1
                    / (1 - t + 1e-4)
                    * (endpoint[:, self.nb_back :, :, :, :] - tmp_noise)
                )

                tmp_noise = tmp_noise + velocity * 1.0 / nb_step
            elif self.parametrization == "noisy":
                noise = self.forward(input_model, x_scalar)

                velocity = (
                    1
                    / (t + 1e-4)
                    * (tmp_noise - noise[:, self.nb_back :, :, :, :])
                )

                tmp_noise = tmp_noise + velocity * 1.0 / nb_step

        return tmp_noise, target_radar_frames, input_radar_frames

    # on epoch end of training
    def on_train_epoch_end(self):
        # generate image
        self.model.eval()

        with torch.no_grad():
            result, target_radar_frames, input_radar_frames = self.generate_one(
                nb_batch=1, nb_step=100
            )

            # reshape first
            self.save_image(result[:, 0, :, :, :], name_append="result")
            self.save_image(target_radar_frames[:, 0, :, :, :], name_append="target")

            self.save_gif(result, name_append="result_gif")
            self.save_gif(target_radar_frames, name_append="target_gif")

            # now we delete all the png files (not the gif)
            for f in glob.glob(self.dir_save + "data/*.png"):
                os.remove(f)

        self.model.train()

    def save_gif(self, result, name_append="result", duration=10):
        nb_frame = result.shape[1]
        file_name_list = []

        for i in range(nb_frame):
            fname = (
                self.dir_save
                + f"data/{name_append}_radar_epoch_{self.current_epoch}_{i}.png"
            )

            radar_image = (
                self.vae.decode(result[:, i, :, :, :] / NORMALIZATION_FACTOR)
                .sample.cpu()
                .numpy()[0, 0, :, :]
            )

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
        radar_image = result

        # decode with vae
        radar_image = (
            self.vae.decode(radar_image / NORMALIZATION_FACTOR)
            .sample.cpu()
            .numpy()[0, 0, :, :]
        )

        fname = (
            self.dir_save + f"data/{name_append}_radar_epoch_{self.current_epoch}.png"
        )

        plt.figure(figsize=(20, 20))
        plt.imshow(radar_image, vmin=-0.5, vmax=2)
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
