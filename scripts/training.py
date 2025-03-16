"""
Script for module training


"""

from meteolibre_model.dataset import (
    MeteoLibreDataset,
    columns_measurements,
)

from meteolibre_model.pl_model import MeteoLibrePLModel

import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


def init_dataset(index_file, dir_index, groundstations_info, groundheight_info):
    dataset = MeteoLibreDataset(
        index_file=index_file,
        dir_index=dir_index,
        groundstations_info=groundstations_info,
        ground_height_image=groundheight_info,
        nb_back_steps=3,
        nb_future_steps=1,
    )

    return dataset


if __name__ == "__main__":
    index_file = "/teamspace/studios/this_studio/data/index.parquet"
    dir_index = "/teamspace/studios/this_studio/data"
    groundstations_info = "/teamspace/studios/this_studio/data/groundstations_filter/total_transformed.parquet"
    groundheight_info = (
        "/teamspace/studios/this_studio/data/reprojected_gebco_32630_500m_padded.npy"
    )

    dataset = init_dataset(
        index_file, dir_index, groundstations_info, groundheight_info
    )

    # For simplicity, use the same dataset for training and validation.
    # In a real scenario, you should have separate datasets.
    train_dataset = dataset
    val_dataset = dataset  # Using same dataset for now

    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_dataloader = DataLoader(
        val_dataset, batch_size=1
    )  # Optional, if you want validation

    model = MeteoLibrePLModel(
        input_channels_ground=len(columns_measurements),
        condition_size=1,
    )

    logger = TensorBoardLogger("tb_logs/", name="g2pt_grid")
    # logger = WandbLogger(project="g2pt_grid")

    trainer = pl.Trainer(
        max_time={"hours": 3},
        logger=logger,
        accumulate_grad_batches=16,
        # fast_dev_run=True,
        # accelerator="cpu", # debug
        gradient_clip_val=1.0,
    )  # fast_dev_run=True for quick debugging

    trainer.fit(
        model, train_dataloader, val_dataloader
    )  # Pass val_dataloader if you have validation step in model

    print("Training finished!")
