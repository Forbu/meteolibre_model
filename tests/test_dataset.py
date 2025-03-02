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
def index_file():
    """Create test fixtures for each test"""
    # Create a temporary directory for test files
    return "/home/adrienbufort/data/index.parquet"

@pytest.fixture
def dir_index():
    """Create test fixtures for each test"""
    # Create a temporary directory for test files
    return "/home/adrienbufort/data"

@pytest.fixture
def groundstations_info():
    """Create test fixtures for each test"""
    # Create a temporary directory for test files
    return "/home/adrienbufort/meteolibre_model/scripts/total_transformed.parquet"

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
    item = dataset[2000]
    
    # becnhmark the speed to get the item
    start_time = datetime.datetime.now()
    for _ in range(10):
        item = dataset[_]
    end_time = datetime.datetime.now()
    print(f"Time to get 10 items: {end_time - start_time}")

    # Check if the item has the expected keys
    assert 'back_0' in item
    assert 'back_1' in item
    assert 'future_0' in item
    assert 'ground_station_image_previous' in item
    assert 'ground_station_image_next' in item

