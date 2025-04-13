"""
Script for module training
"""

from meteolibre_model.vae.pl_model_uk_vae_3d import VAEMeteoLibrePLModelGrid
from meteolibre_model.dataset_uk_dm import TFDataset

from meteolibre_model.dataset_cutting_grid import (
    columns_measurements,
)

from safetensors.torch import save_file
from huggingface_hub import HfApi

import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger

from torch.utils.data import DataLoader

import torch
torch.set_float32_matmul_precision('medium')

from lightning.pytorch.callbacks import ModelCheckpoint

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

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True,) #num_workers=8)
    val_dataloader = DataLoader(
        val_dataset, batch_size=1, shuffle=True
    )  # Optional, if you want validation

    model = VAEMeteoLibrePLModelGrid(
        test_dataloader=val_dataloader,
    )

    # logger = TensorBoardLogger("tb_logs/", name="g2pt_grid")
    logger = WandbLogger(project="meteolibre_model_vae")

    # model checkpoint 
    callback = ModelCheckpoint(every_n_epochs=3, save_last=True, dirpath="models/finetune_vae_3d_v0/")


    trainer = pl.Trainer(
        max_time={"hours": 10},
        logger=logger,
        accumulate_grad_batches=2,
        #fast_dev_run=True,
        #accelerator="cpu", # debug
        callbacks=[callback],
        gradient_clip_val=1.0,
        log_every_n_steps=5,
        #enable_checkpointing=False,
    )  # fast_dev_run=True for quick debugging

    trainer.fit(
        model, train_dataloader, val_dataloader
    )  # Pass val_dataloader if you have validation step in model

    print("Training finished!")

    # Save the model in safetensors format
    save_file(model.model.state_dict(), "diffusion_pytorch_model.safetensors")

    #torch.save(model.model.state_dict(), "model_vae.pt")


    # push file to hub
    api = HfApi()
    api.upload_file(
        path_or_fileobj="diffusion_pytorch_model.safetensors",
        path_in_repo="weights_vae_3d/diffusion_pytorch_model.safetensors",
        repo_id="Forbu14/meteolibre",
        repo_type="model",
    )