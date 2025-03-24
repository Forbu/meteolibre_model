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

import segmentation_models_pytorch as smp
import torch.nn as nn

from meteolibre_model.utils import FilmLayer

class UnetFilmModel(nn.Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        condition_size,
        encoder_name="efficientnet-b3",
    ):
        super().__init__()

        self.model = smp.Unet(
            encoder_name=encoder_name,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            in_channels=input_channels,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=output_channels,  # model output channels (number of classes in your dataset)
        )

        self.film1 = FilmLayer(num_features=input_channels, condition_size=condition_size)
        self.film2 = FilmLayer(output_channels, condition_size=condition_size)

    def forward(self, x_image, x_scalar):
        """
        Args:
            x_image (torch.Tensor): Input image tensor (B, H, W, C_in).
            x_scalar (torch.Tensor): Scalar input tensor (B, condition_size).

        Returns:
            torch.Tensor: Output image tensor (B, H, W, C_out).
        """
        x_image = x_image.permute(0, 3, 1, 2)

        out = self.film1(x_image, x_scalar)
        out = self.model(x_image)
        out = self.film2(out, x_scalar)

        out = out.permute(0, 2, 3, 1)

        return out
