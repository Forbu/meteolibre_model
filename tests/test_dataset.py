"""
In this file we will test the dataset class
"""

import os
import pytest
import tempfile
import datetime
import numpy as np
import h5py
import torch

import matplotlib.pyplot as plt


from meteolibre_model.dataset import (
    MeteoLibreDataset,
    transform_groundstation_data_into_image,
)


@pytest.fixture
def dir_index():
    """Create test fixtures for each test"""
    # Create a temporary directory for test files
    return "/home/adrienbufort/meteolibre_dataset/data"


@pytest.fixture
def index_file(dir_index):
    """Create test fixtures for each test"""
    # Create a temporary directory for test files
    return os.path.join(dir_index, "index.parquet")


@pytest.fixture
def groundstations_info(dir_index):
    """Create test fixtures for each test"""
    # Create a temporary directory for test files
    return os.path.join(dir_index, "groundstations_filter/total_transformed.parquet")


@pytest.fixture
def groundheight_info(dir_index):
    """Create test fixtures for each test"""
    # Create a temporary directory for test files
    return os.path.join(dir_index, "assets/reprojected_gebco_32630_500m_padded.npy")


def test_getitem(index_file, dir_index, groundstations_info, groundheight_info):
    """Test __getitem__ method"""
    # Setup mock for h5py.File to return a dict-like object with a 'data' key

    dataset = MeteoLibreDataset(
        index_file=index_file,
        dir_index=dir_index,
        groundstations_info=groundstations_info,
        ground_height_image=groundheight_info,
        nb_back_steps=2,
        nb_future_steps=1,
    )

    # Get an item
    item = dataset[5000]

    print(item.keys())

    keys_toplot = [
        "back_0",
        "back_1",
        "future_0",
        # "ground_station_image_previous",
        # "ground_station_image_next",
        "mask_previous",
        "mask_next",
        "ground_height_image",
    ]


    for key in keys_toplot:
        plt.figure()

        array = item[key]

        if "mask" not in key:
            array[array == array.max()] = 0
            
            # print number of non zero element

            
        else:
            array = array[:,:, 0]
            
            array = np.int8(array)
            
            # we want to extand the point with 1 value (with convolution)
            # import numpy as np
            from scipy.signal import convolve2d
            array = convolve2d(array, np.ones((3,3)), mode='same')
            array = convolve2d(array, np.ones((3,3)), mode='same')
            array = convolve2d(array, np.ones((3,3)), mode='same')
            array = convolve2d(array, np.ones((3,3)), mode='same')
            
            # clip to 1
            array = np.clip(array, 0, 1)

        plt.imshow(array)
        plt.title(key)
        plt.colorbar()

        plt.savefig(f"/home/adrienbufort/meteolibre_model/tests/{key}.png")

    exit()

    # # becnhmark the speed to get the item
    # start_time = datetime.datetime.now()
    # for _ in range(10):
    #     item = dataset[_]
    # end_time = datetime.datetime.now()
    # print(f"Time to get 10 items: {end_time - start_time}")

    # Check if the item has the expected keys
    assert "back_0" in item
    assert "back_1" in item
    assert "future_0" in item
    assert "ground_station_image_previous" in item
    assert "ground_station_image_next" in item
