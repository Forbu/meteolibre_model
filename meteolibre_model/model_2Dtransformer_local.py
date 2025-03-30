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

from meteolibre_model.model_dit import DiT
import torch.nn as nn
import torch
import einops


class TransfomerFilmModel(nn.Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        condition_size,
        patch_size=2,
        subimage_size=32,
        full_image_size=256,
        hidden_size=384,
    ):
        super().__init__()

        self.dit_block = torch.nn.ModuleList()

    
        self.model = (
            DiT(
                depth=12,
                hidden_size=hidden_size,
                patch_size=2,
                num_heads=6,
                input_size=subimage_size,
                in_channels=hidden_size // 8,
                out_channels=hidden_size // 8,
            )
        )

        self.condition_size = condition_size
        self.subimage_size = subimage_size
        self.full_image_size = full_image_size

        # projection to hidden size for scalar value
        self.mlp = nn.Sequential(
            nn.Linear(condition_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

        # projection to hidden size for image
        self.image_mlp = nn.Sequential(
            nn.Linear(input_channels, hidden_size // 8, bias=True),
        )

        # head layer (project back
        self.head = nn.Sequential(
            nn.Linear(hidden_size // 8, hidden_size // 8, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size // 8, output_channels, bias=True),
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
        batch_size = x_image.shape[0]

        x_scalar = self.mlp(x_scalar)
        x_image = self.image_mlp(x_image)

        x_image = x_image.permute(0, 3, 1, 2)  # (B, C, H, W)

        x_image = self.pass_through(x_image, x_scalar, self.model)

        #for block in self.dit_block:
            # x image is of size (B, C, H, W)
            # we also want to create a padded image of shape  (B, C, H + subimage_size, W + subimage_size)
            # with + subimage // 2 at the top and bottom and + subimage // 2 at the left and right
            # x_image_padded = torch.nn.functional.pad(
            #     x_image,
            #     (0, 0, self.subimage_size // 2, self.subimage_size // 2),
            #     mode="constant",
            # )

            # x_image_padded = torch.nn.functional.pad(
            #     x_image_padded,
            #     (self.subimage_size // 2, self.subimage_size // 2, 0, 0),
            #     mode="constant",
            #     value=0,
            # )

            # x_image_tmp = self.pass_through(x_image, x_scalar, block)
            #x_image_padded_tmp = self.pass_through(x_image_padded, x_scalar, block)

            # x_image = x_image_tmp
            #     + x_image_padded_tmp[
            #         :,
            #         :,
            #         self.subimage_size // 2 : self.subimage_size // 2
            #         + x_image.shape[2],
            #         self.subimage_size // 2 : self.subimage_size // 2
            #         + x_image.shape[3],
            #     ]
            # )

        x_image = x_image.permute(0, 2, 3, 1)

        # header
        x_image = self.head(x_image)

        return x_image

    def pass_through(self, x_image, x_scalar, block):
        batch_size = x_image.shape[0]
        full_image_size = x_image.shape[2]

        subimage_setup = self.transform_to_subimage(
            x_image
        )  # (B, C, H/subimage_size, W/subimage_size, subimage_size, subimage_size)

        x_scalar = x_scalar.unsqueeze(1).unsqueeze(1)
        x_scalar = x_scalar.repeat(
            1,
            full_image_size // self.subimage_size,
            full_image_size // self.subimage_size,
            1,
        )
        x_scalar = einops.rearrange(x_scalar, "b h w c -> (b h w) c")

        subimage_setup = block(subimage_setup, x_scalar)

        x_image = self.transform_from_subimage(
            subimage_setup, batch_size, full_image_size
        )

        return x_image

    def transform_to_subimage(self, x_image):
        x_image = x_image = einops.rearrange(
            x_image,
            "b c (h s1) (w s2) -> (b h w) c s1 s2",
            s1=self.subimage_size,
            s2=self.subimage_size,
        )

        return x_image

    def transform_from_subimage(self, x_image, batch_size, full_image_size):
        x_image = einops.rearrange(
            x_image,
            "(b h w) c s1 s2 -> b c (h s1) (w s2)",
            b=int(batch_size),
            h=full_image_size // self.subimage_size,
            w=full_image_size // self.subimage_size,
        )

        return x_image
