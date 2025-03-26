"""
meteolibre_model/meteolibre_model/pl_model.py
"""

import torch
import torch.nn as nn
import lightning.pytorch as pl
from meteolibre_model.model_unet import UnetFilmModel
from meteolibre_model.test_dataset_uk_dm import TFDataset
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

import wandb


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
        learning_rate=1e-2,
        nb_back=3,
        nb_future=1,
        shape_image=512,
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
        self.model = UnetFilmModel(
            nb_back + nb_future,
            nb_future,
            condition_size,
        )
        self.learning_rate = learning_rate
        self.criterion = nn.MSELoss(reduction="none")  # Rectified Flow uses MSE loss

        self.test_dataloader = test_dataloader

        self.dir_save = dir_save

        self.nb_back = nb_back
        self.nb_future = nb_future

        self.shape_image = shape_image

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

        # Prior sample (simple Gaussian noise) - you can refine this prior
        prior_image = torch.randn_like(batch["target_radar_frames"])

        # Time variable for Rectified Flow - sample uniformly
        t = torch.rand(batch["target_radar_frames"].shape[0], 1, 1, 1).type_as(
            batch["target_radar_frames"]
        )  # (B, 1)

        # we create a scalar value to condition the model on time stamp
        # and hours
        x_hour = batch["hour"].clone().detach().float().unsqueeze(1)  # (B, 1)
        x_minute = batch["minute"].clone().detach().float().unsqueeze(1)  # (B, 1)

        # Simple scalar condition: hour of the day and minutes of the day. You might want to expand this.
        x_scalar = torch.cat([x_hour, x_minute, t], dim=1)

        # Interpolate between prior and data to get x_t
        x_t = t * batch["target_radar_frames"] + (1 - t) * prior_image

        # concat x_t with x_image_back and x_ground_station_image_previous
        input_model = torch.cat([x_t, batch["input_radar_frames"]], dim=-1)

        # Predict velocity field v_t using the model
        end_t_predicted = self.forward(input_model, x_scalar)

        # coefficient of ponderation for endpoint parametrization
        w_t = t / (1 - t)
        w_t = torch.clamp(w_t, min=0.05, max=2.0)

        # Loss is MSE between predicted and target velocity fields
        loss = self.criterion(end_t_predicted, batch["target_radar_frames"])
        loss = w_t * loss  # ponderate loss

        loss = loss.mean()

        # Log the loss
        self.log("train_loss", loss)

        return loss

    def configure_optimizers(self):
        """
        Configures the optimizer for training.

        Returns:
            torch.optim.Optimizer: Adam optimizer.
        """
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer

    @torch.no_grad()
    def generate_one(self, nb_batch=1, nb_step=100):
        # Assuming batch is a dictionary returned by TFDataset
        # we first generate a random noise
        for batch in self.test_dataloader:
            break

        # convert to device
        batch = {k: v.to(self.device) for k, v in batch.items()}

        x_image_future = batch["target_radar_frames"]
        x_image_back = batch["input_radar_frames"]


        tmp_noise = torch.randn_like(x_image_future).to(self.device)

        for i in range(nb_step):
            # concat x_t with x_image_back and x_ground_station_image_previous
            input_model = torch.cat([tmp_noise, x_image_back], dim=-1)

            t = i * 1.0 / nb_step

            x_scalar = torch.cat(
                [
                    batch["hour"].clone().detach().float().unsqueeze(1),
                    batch["minute"].clone().detach().float().unsqueeze(1),
                    t,
                ],
                dim=1,
            )

            endpoint = self.forward(input_model, x_scalar)

            # velocity field
            velocity = 1 / (1 - t + 1e-4) * (endpoint - tmp_noise)

            # addinf the velocity to the noise
            tmp_noise = tmp_noise + 1.0 / nb_step * velocity

        return tmp_noise, x_image_future, x_image_back[:, :, :, [-1]]

    # on epoch end of training
    def on_train_epoch_end(self):
        # generate image
        self.eval()
        result, x_image_future, x_image_back = self.generate_one(
            nb_batch=1, nb_step=100
        )

        self.save_image(result, name_append="result")
        self.save_image(x_image_future, name_append="future")
        self.save_image(x_image_back, name_append="previous")

        self.train()

    def save_image(self, result, name_append="result"):
        radar_image = result[:, :, :, -1].cpu().numpy()

        radar_image = radar_image[0]

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
