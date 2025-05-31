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


from meteolibre_model.datasets.dataset_meteofrance import (
    MeteoLibreDataset,
)

def test_dataset():
    dataset = MeteoLibreDataset(directory="/teamspace/studios/this_studio/data/data/hf_dataset/")

    assert isinstance(dataset, MeteoLibreDataset)

