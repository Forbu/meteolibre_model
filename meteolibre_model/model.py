"""
model.py

Simple model for MeteoLibre that takes the image with multiple channels and outputs the next image with multiple channels.
Take also scalar element (matching flow time and hour of the day) as input.
The model is a simple CNN with some conv layers and some linear layers.
"""

"""
model.py

Simple model for MeteoLibre that takes the image with multiple channels and outputs the next image with multiple channels.
Take also scalar element (matching flow time and hour of the day) as input.
The model is a simple CNN with some conv layers and also film layers for the scalar input.
"""

import torch.nn as nn

from meteolibre_model.utils import FilmLayer

class SimpleConvFilmModel(nn.Module):
    def __init__(self, input_channels, output_channels, condition_size):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.film1 = FilmLayer(32, condition_size)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.film2 = FilmLayer(64, condition_size)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(64, output_channels, kernel_size=3, padding=1)

    def forward(self, x_image, x_scalar):
        """
        Args:
            x_image (torch.Tensor): Input image tensor (B, H, W, C_in).
            x_scalar (torch.Tensor): Scalar input tensor (B, condition_size).

        Returns:
            torch.Tensor: Output image tensor (B, H, W, C_out).
        """
        x_image = x_image.permute(0, 3, 1, 2)

        out = self.conv1(x_image)
        out = self.film1(out, x_scalar)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.film2(out, x_scalar)
        out = self.relu2(out)
        out = self.conv3(out)

        out = out.permute(0, 2, 3, 1)

        return out

