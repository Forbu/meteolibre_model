"""
Script for module training


"""

from meteolibre_model.pl_model_grid import MeteoLibrePLModelGrid


from meteolibre_model.dataset_cutting_grid import MeteoLibreDatasetPartialGrid


from meteolibre_model.dataset_cutting_grid import (
    columns_measurements,
)

from meteolibre_model.pl_model_grid import MeteoLibrePLModelGrid

import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


def init_dataset(index_file, dir_index, groundheight_info):
    dataset = MeteoLibreDatasetPartialGrid(
        index_file=index_file,
        dir_index=dir_index,
        ground_height_image=groundheight_info,
        nb_back_steps=3,
        nb_future_steps=1,
    )

    return dataset


if __name__ == "__main__":
    index_file = "/teamspace/studios/this_studio/data/index.parquet"
    dir_index = "/teamspace/studios/this_studio/data"
    groundheight_info = (
        "/teamspace/studios/this_studio/data/reprojected_gebco_32630_500m_padded.npy"
    )

    dataset = init_dataset(
        index_file, dir_index, groundheight_info
    )

    # For simplicity, use the same dataset for training and validation.
    # In a real scenario, you should have separate datasets.
    train_dataset = dataset
    val_dataset = dataset  # Using same dataset for now

    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=8)
    val_dataloader = DataLoader(
        val_dataset, batch_size=1, shuffle=True
    )  # Optional, if you want validation

    model = MeteoLibrePLModelGrid(
        condition_size=2,
        test_dataloader=val_dataloader,
        scale_factor_reduction=1,
    )

    # logger = TensorBoardLogger("tb_logs/", name="g2pt_grid")
    logger = WandbLogger(name="meteolibre_model", project="meteolibre_model")

    trainer = pl.Trainer(
        max_time={"hours": 3},
        logger=logger,
        accumulate_grad_batches=8,
        # fast_dev_run=True,
        # accelerator="cpu", # debug
        gradient_clip_val=1.0,
        log_every_n_steps=5,
    )  # fast_dev_run=True for quick debugging

    trainer.fit(
        model, train_dataloader, val_dataloader
    )  # Pass val_dataloader if you have validation step in model

    # Save the model
    trainer.save_checkpoint(f"meteolibre_model_{trainer.current_epoch}.ckpt")

    print("Training finished!")
