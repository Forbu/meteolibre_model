import torch
from meteolibre_model.dit.dit_core import DiTCore, FinalLayer


def test_final_layer():
    """
    Tests the FinalLayer's output shape.
    """
    hidden_size = 384
    patch_size = 2
    out_channels = 16
    final_layer = FinalLayer(hidden_size, patch_size, out_channels)

    N, T = 4, 256
    x = torch.randn(N, T, hidden_size)

    output = final_layer(x)

    expected_shape = (N, T, patch_size * patch_size * out_channels)
    assert output.shape == expected_shape


def test_dit_core_forward_pass():
    """
    Tests the forward pass of the DiTCore model, checking the output shape.
    """
    hidden_size = 32
    patch_size = 2
    out_channels = 16
    in_channels = 16  # must match PatchEmbed in_chans
    nb_temporals = 8

    model = DiTCore(
        nb_temporals=nb_temporals,
        hidden_size=hidden_size,
        patch_size=patch_size,
        out_channels=out_channels,
        in_channels=in_channels,
    )

    batch_size = 2

    height = 32  # must match PatchEmbed img_size
    width = 32  # must match PatchEmbed img_size

    x = torch.randn(batch_size, in_channels, nb_temporals, height, width)
    scalar_input = torch.randn(batch_size, hidden_size)

    output = model(x, scalar_input)

    expected_shape = (batch_size, out_channels, nb_temporals, height, width)
    assert output.shape == expected_shape


def test_unpatchify():
    """
    Tests the unpatchify method to ensure it correctly reconstructs the image.
    """
    model = DiTCore(nb_temporals=8)

    N = 8  # batch_size * nb_temporals
    T = 256  # h * w (16*16)
    patch_size = model.x_embedder.patch_size[0]
    out_channels = model.out_channels

    x = torch.randn(N, T, patch_size * patch_size * out_channels)

    imgs = model.unpatchify(x)

    h = w = int(T**0.5) * patch_size
    expected_shape = (N, out_channels, h, w)
    assert imgs.shape == expected_shape
