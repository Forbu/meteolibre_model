"""
In this file we will test the dataset class
"""

import os
import pytest
import tempfile
import datetime
import numpy as np
import polars as pl
import h5py
import torch
from unittest.mock import patch, MagicMock

from meteolibre_model.dataset import MeteoLibreDataset, transform_groundstation_data_into_image


@pytest.fixture
def dataset_fixture():
    """Create test fixtures for each test"""
    # Create a temporary directory for test files
    
    pass


def test_getitem(index_file, dir_index, groundstations_info):
    """Test __getitem__ method"""
    # Setup mock for h5py.File to return a dict-like object with a 'data' key

    dataset = MeteoLibreDataset(
        index_file=index_file,
        dir_index=dir_index,
        groundstations_info=groundstations_info,
        nb_back_steps=2,
        nb_future_steps=1
    )
    
    # Get an item
    item = dataset[0]
    
    # Check if the item has the expected keys
    assert 'back_0' in item
    assert 'back_1' in item
    assert 'future_0' in item
    assert 'ground_station_image_previous' in item
    assert 'ground_station_image_next' in item

