"""
Script for module training


"""

from meteolibre_model.pl_model_uk import MeteoLibrePLModelGrid
from meteolibre_model.dataset_uk_dm import TFDataset

from meteolibre_model.dataset_cutting_grid import (
    columns_measurements,
)

import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger

from torch.utils.data import DataLoader

import torch
torch.set_float32_matmul_precision('medium')

def init_dataset():
    dataset = TFDataset(
        "train",
    )

    return dataset


if __name__ == "__main__":

    dataset = init_dataset(
    )

    # For simplicity, use the same dataset for training and validation.
    # In a real scenario, you should have separate datasets.
    train_dataset = dataset
    val_dataset = dataset  # Using same dataset for now

    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True,) #num_workers=8)
    val_dataloader = DataLoader(
        val_dataset, batch_size=1, shuffle=True
    )  # Optional, if you want validation

    model = MeteoLibrePLModelGrid(
        condition_size=3,
        test_dataloader=val_dataloader,
        nb_back=4,
        nb_future=4,
        loss_type="mse",
        parametrization="noisy",
    )

    # logger = TensorBoardLogger("tb_logs/", name="g2pt_grid")
    logger = WandbLogger(project="meteolibre_model")

    trainer = pl.Trainer(
        max_time={"hours": 3},
        logger=logger,
        accumulate_grad_batches=8,
        #fast_dev_run=True,
        #accelerator="cpu", # debug
        gradient_clip_val=1.0,
        log_every_n_steps=5,
    )  # fast_dev_run=True for quick debugging

    trainer.fit(
        model, train_dataloader, val_dataloader
    )  # Pass val_dataloader if you have validation step in model

    print("Training finished!")
