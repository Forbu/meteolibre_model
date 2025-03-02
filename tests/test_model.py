import pytest
import torch
from meteolibre_model.model import SimpleConvFilmModel

def test_output_shape():
    input_channels = 3
    output_channels = 2
    condition_size = 4
    batch_size = 2
    height = 64
    width = 64

    model = SimpleConvFilmModel(input_channels, output_channels, condition_size)
    input_image = torch.randn(batch_size, input_channels, height, width)
    input_scalar = torch.randn(batch_size, condition_size)
    output = model(input_image, input_scalar)

    assert output.shape == (batch_size, output_channels, height, width), "Output shape is incorrect"

def test_forward_pass_no_error():
    input_channels = 3
    output_channels = 2
    condition_size = 4
    batch_size = 2
    height = 64
    width = 64

    model = SimpleConvFilmModel(input_channels, output_channels, condition_size)
    input_image = torch.randn(batch_size, input_channels, height, width)
    input_scalar = torch.randn(batch_size, condition_size)

    try:
        model(input_image, input_scalar)
    except Exception as e:
        pytest.fail(f"Forward pass raised an exception: {e}")

def test_different_input_sizes():
    input_channels = 1
    output_channels = 1
    condition_size = 2
    batch_size = 4
    height = 32
    width = 32

    model = SimpleConvFilmModel(input_channels, output_channels, condition_size)
    input_image = torch.randn(batch_size, input_channels, height, width)
    input_scalar = torch.randn(batch_size, condition_size)

    try:
        model(input_image, input_scalar)
        output = model(input_image, input_scalar)
        assert output.shape == (batch_size, output_channels, height, width), "Output shape is incorrect for different input sizes"
    except Exception as e:
        pytest.fail(f"Forward pass with different input sizes raised an exception: {e}")
