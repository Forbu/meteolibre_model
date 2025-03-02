"""
meteolibre_model/meteolibre_model/pl_model.py
"""
import torch
import torch.nn as nn
import pytorch_lightning as pl
from meteolibre_model.model import SimpleConvFilmModel
from meteolibre_model.dataset import MeteoLibreDataset
from torch.utils.data import DataLoader

class MeteoLibrePLModel(pl.LightningModule):
    def __init__(self, input_channels_ground, output_channels, condition_size, learning_rate=1e-3, nb_back=3, nb_future=1, nb_hidden=16):
        super().__init__()
        self.model = SimpleConvFilmModel(nb_hidden + nb_back, output_channels, condition_size)
        self.learning_rate = learning_rate
        self.criterion = nn.MSELoss(reduction='none') # Rectified Flow uses MSE loss
        
        self.nb_back = nb_back
        self.nb_future = nb_future
        self.hidden_size = nb_hidden
        self.input_channels_ground = input_channels_ground
        
        # embedding for non know value (radar)
        self.embedding_non_know_ground_station = nn.Parameter(torch.randn(1, 1, 1, 16))
        
        # simple projection of the groundstation image to the hidden dimension
        self.projection_ground_station = nn.Linear(input_channels_ground, nb_hidden)

    def forward(self, x_image, x_scalar):
        return self.model(x_image, x_scalar)

    def training_step(self, batch, batch_idx):
        # Assuming batch is a dictionary returned by MeteoLibreDataset
        # and contains 'back_0', 'future_0', 'hour' keys

        img_batck_list = []

        for i in range(self.nb_back):
            x_image_back = torch.tensor(batch['back_{i}'], dtype=torch.float32) # (B, H, W, C)
            img_batck_list.append(x_image_back)
        
        # we project the groundstation (ground_station_image_previous) to hidden dimension
        x_ground_station_image_previous = batch['ground_station_image_previous'] # (B, H, W, C)
        x_ground_station_image_previous = self.projection_ground_station(x_ground_station_image_previous) # (B, H, W, hidden_size)
        
        mask_previous = batch['mask_previous'] # (B, H, W)
        mask_previous = mask_previous.unsqueeze(-1) # (B, H, W, 1)
        x_ground_station_image_previous = x_ground_station_image_previous * mask_previous + self.embedding_non_know_ground_station * (1 - mask_previous) # (B, H, W, hidden_size)
        
        # Concatenate all back images along the channel dimension
        x_image_back = torch.cat(img_batck_list, dim=1) # (B, H, W, C*nb_back)
        
        x_image_future = torch.tensor(batch['future_0'], dtype=torch.float32) # (B, C, H, W)
        
        x_hour = torch.tensor(batch['hour'], dtype=torch.float32).unsqueeze(1) # (B, 1)

        # Simple scalar condition: hour of the day. You might want to expand this.
        x_scalar = x_hour / 24.0 # Normalize hour to [0, 1]

        # Prior sample (simple Gaussian noise) - you can refine this prior
        prior_image = torch.randn_like(x_image_back)

        # Time variable for Rectified Flow - sample uniformly
        t = torch.rand(x_image_back.shape[0], 1).type_as(x_image_back) # (B, 1)

        # Interpolate between prior and data to get x_t
        x_t = t * x_image_future + (1 - t) * prior_image

        # Predict velocity field v_t using the model
        v_t_predicted = self.forward(x_t, x_scalar)

        # Target velocity field in Rectified Flow is simply (data - prior)
        v_target = x_image_future - prior_image

        # Loss is MSE between predicted and target velocity fields
        loss = self.criterion(v_t_predicted, v_target)
        
        # we mask the loss for the ground station
        mask_loss = batch['mask_next'] # (B, H, W, input_channels_ground)
        
        loss_ground = (loss[:, :, :, :self.input_channels_radar] * mask_loss).sum() / mask_loss.sum()
        loss_radar = loss[:, :, :, -1].mean()

        loss = loss_ground + loss_radar

        # Log the loss
        self.log('train_loss', loss)
        self.log('train_loss_ground', loss_ground)
        self.log('train_loss_radar', loss_radar)
        
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
