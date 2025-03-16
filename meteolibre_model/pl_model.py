"""
meteolibre_model/meteolibre_model/pl_model.py
"""

import torch
import torch.nn as nn
import lightning.pytorch as pl
from meteolibre_model.model import SimpleConvFilmModel
from meteolibre_model.dataset import MeteoLibreDataset
from torch.utils.data import DataLoader


class MeteoLibrePLModel(pl.LightningModule):
    """
    PyTorch Lightning module for the MeteoLibre model.

    This class wraps the SimpleConvFilmModel and integrates it with PyTorch Lightning
    for training and evaluation. It handles data loading, optimization, and defines
    the training step using a Rectified Flow approach with MSE loss.
    """

    def __init__(
        self,
        input_channels_ground,
        condition_size,
        learning_rate=1e-3,
        nb_back=3,
        nb_future=1,
        nb_hidden=16,
        scale_factor_reduction=2,
        shape_image=3472,
    ):
        """
        Initialize the MeteoLibrePLModel.

        Args:
            input_channels_ground (int): Number of input channels for the ground station image.
            condition_size (int): Size of the conditioning vector.
            learning_rate (float, optional): Learning rate for the optimizer. Defaults to 1e-3.
            nb_back (int, optional): Number of past frames to use as input. Defaults to 3.
            nb_future (int, optional): Number of future frames to predict. Defaults to 1.
            nb_hidden (int, optional): Number of hidden channels in the model. Defaults to 16.
        """
        super().__init__()
        self.model = SimpleConvFilmModel(
            2 * input_channels_ground + nb_back + nb_future,
            input_channels_ground + nb_future,
            condition_size,
        )
        self.learning_rate = learning_rate
        self.criterion = nn.MSELoss(reduction="none")  # Rectified Flow uses MSE loss

        self.nb_back = nb_back
        self.nb_future = nb_future
        self.hidden_size = nb_hidden
        self.input_channels_ground = input_channels_ground
        self.shape_image = shape_image

        # embedding for non know value (radar)
        self.embedding_non_know_ground_station = nn.Parameter(
            torch.randn(1, 1, 1, input_channels_ground)
        )

        # batchnorm
        self.batchnorm_radar = torch.nn.BatchNorm2d(
            input_channels_ground, momentum=None
        )

        self.scale_factor_reduction = scale_factor_reduction
        self.maxpool = nn.MaxPool2d(
            kernel_size=scale_factor_reduction, stride=scale_factor_reduction
        )

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
                          expected to be returned by MeteoLibreDataset.
            batch_idx (int): Index of the current batch.

        Returns:
            torch.Tensor: Training loss for the batch.
        """
        # Assuming batch is a dictionary returned by MeteoLibreDataset
        # and contains 'back_0', 'future_0', 'hour' keys

        img_batck_list = []

        for i in range(self.nb_back):
            x_image_back = batch[f"back_{i}"].clone().detach().float()# (B, H, W, C)
            img_batck_list.append(x_image_back)

        mask_previous = batch["mask_previous"].clone().detach()  # (B, H, W, C)
        mask_future = batch["mask_next"].clone().detach()  # (B, H, W, C)

        x_image_future = batch["future_0"].clone().detach().float() # (B, H, W, C)

        x_hour = batch["hour"].clone().detach().float().unsqueeze(1)  # (B, 1)
        x_minute = batch["minute"].clone().detach().float().unsqueeze(1)  # (B, 1)

        # Simple scalar condition: hour of the day and minutes of the day. You might want to expand this.
        x_scalar = torch.cat([x_hour, x_minute], dim=1)

        # we project the groundstation (ground_station_image_previous) to hidden dimension
        x_ground_station_image_previous = batch[
            "ground_station_image_previous"
        ]  # (B, H, W, C)

        x_ground_station_image_previous = (
            x_ground_station_image_previous * mask_previous
            + self.embedding_non_know_ground_station * (1 - mask_previous)
        )  # (B, H, W, hidden_size)

        # we project the groundstation (ground_station_image_previous) to hidden dimension
        x_ground_station_image_future = batch[
            "ground_station_image_next"
        ]  # (B, H, W, C)

        x_image_future = torch.cat(
            [x_ground_station_image_future, x_image_future.unsqueeze(3)], dim=-1
        )

        # Concatenate all back images along the channel dimension
        x_image_back = torch.stack(img_batck_list, dim=-1)  # (B, H, W, C*nb_back)

        (
            x_image_future,
            x_image_back,
            x_ground_station_image_previous,
            x_ground_station_image_future,
            mask_future,
            mask_previous,
        ) = self.pooling_operation(
            x_image_future,
            x_image_back,
            x_ground_station_image_previous,
            x_ground_station_image_future,
            mask_future,
            mask_previous,
        )

        # Prior sample (simple Gaussian noise) - you can refine this prior
        prior_image = torch.randn_like(x_image_future)

        # Time variable for Rectified Flow - sample uniformly
        t = torch.rand(x_image_future.shape[0], 1, 1, 1).type_as(
            x_image_future
        )  # (B, 1)

        # Interpolate between prior and data to get x_t
        x_t = t * x_image_future + (1 - t) * prior_image

        # concat x_t with x_image_back and x_ground_station_image_previous
        input_model = torch.cat(
            [x_t, x_image_back, x_ground_station_image_previous], dim=-1
        )

        # Predict velocity field v_t using the model
        v_t_predicted = self.forward(input_model, x_scalar)

        # Target velocity field in Rectified Flow is simply (data - prior)
        v_target = x_image_future - prior_image

        # Loss is MSE between predicted and target velocity fields
        loss = self.criterion(v_t_predicted, v_target)

        loss_ground = (
            loss[:, :, :, : self.input_channels_ground] * mask_future
        ).sum() / mask_future.sum()
        loss_radar = loss[:, :, :, -1].mean()

        loss = loss_ground + loss_radar

        # Log the loss
        self.log("train_loss", loss)
        self.log("train_loss_ground", loss_ground)
        self.log("train_loss_radar", loss_radar)

        return loss

    def configure_optimizers(self):
        """
        Configures the optimizer for training.

        Returns:
            torch.optim.Optimizer: Adam optimizer.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def generate_one(self, batch, nb_batch=1, nb_step=100):
        # Assuming batch is a dictionary returned by MeteoLibreDataset
        # we first generate a random noise
        noise = torch.randn(nb_batch, 1, 1, 1).type_as(batch["future_0"])


    def pooling_operation(
        self,
        x_image_future,
        x_image_back,
        x_ground_station_image_previous,
        x_ground_station_image_future,
        mask_future,
        mask_previous,
    ):
        # Apply max pooling to image inputs and masks
        x_t_pooled = self.maxpool(x_image_future.permute(0, 3, 1, 2)).permute(
            0, 2, 3, 1
        )  # (B, H/scale, W/scale, C+1)
        x_image_back_pooled = self.maxpool(x_image_back.permute(0, 3, 1, 2)).permute(
            0, 2, 3, 1
        )  # (B, H/scale, W/scale, C*nb_back)
        x_ground_station_image_previous_pooled = self.maxpool(
            x_ground_station_image_previous.permute(0, 3, 1, 2)
        ).permute(0, 2, 3, 1)  # (B, H/scale, W/scale, C)
        x_ground_station_image_future_pooled = self.maxpool(
            x_ground_station_image_future.permute(0, 3, 1, 2)
        ).permute(0, 2, 3, 1)  # (B, H/scale, W/scale, C)
        mask_future_pooled = self.maxpool(
            mask_future.permute(0, 3, 1, 2).float()
        ).permute(0, 2, 3, 1)  # (B, H/scale, W/scale, C)
        mask_previous_pooled = self.maxpool(
            mask_previous.permute(0, 3, 1, 2).float()
        ).permute(0, 2, 3, 1)  # (B, H/scale, W/scale, C)
        return (
            x_t_pooled,
            x_image_back_pooled,
            x_ground_station_image_previous_pooled,
            x_ground_station_image_future_pooled,
            mask_future_pooled,
            mask_previous_pooled,
        )
