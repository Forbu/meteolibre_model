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

from meteolibre_model.model_dit_3d import DiT_3d
import torch.nn as nn


class TransfomerFilmModel(nn.Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        condition_size,
        patch_size=2,
        hidden_size=384,
    ):
        super().__init__()

        self.hidden_size = hidden_size

        self.model = DiT_3d(
            depth=12,
            hidden_size=hidden_size,
            patch_size=patch_size,
            num_heads=6,
            input_size=32,
            in_channels=input_channels,
            out_channels=output_channels,
        )

        self.condition_size = condition_size

        # projection to hidden size
        self.mlp = nn.Sequential(
            nn.Linear(condition_size, self.hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(self.hidden_size, self.hidden_size, bias=True),
        )

        self.output_channels = output_channels

    def forward(self, x_image, x_scalar):
        """
        Args:
            x_image (torch.Tensor): Input image tensor (B, H, W, C_in).
            x_scalar (torch.Tensor): Scalar input tensor (B, condition_size).

        Returns:
            torch.Tensor: Output image tensor (B, H, W, C_out).
        """
        x_image = x_image.permute(0, 3, 1, 2)

        # image is of size (B, C, H, W)
        # now we want to create patch
        x_scalar = self.mlp(x_scalar)
        out = self.model(x_image, x_scalar)

        out = out.permute(0, 2, 3, 1)

        return out
