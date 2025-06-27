"""
meteolibre_model/meteolibre_model/pl_model.py
"""

import os
import glob
import wandb
import matplotlib.pyplot as plt
from PIL import Image

from mpmath.libmp import prec_to_dps

import torch
import torch.nn as nn
import einops
from torch.optim import optimizer

import lightning.pytorch as pl

from heavyball import ForeachSOAP


NORMALIZATION_FACTOR = 1. #0.769

