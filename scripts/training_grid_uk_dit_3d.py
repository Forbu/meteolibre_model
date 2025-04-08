"""
Script for module training


"""

from meteolibre_model.pl_model_uk_dit_3d import MeteoLibrePLModelGrid
from meteolibre_model.dataset_uk_dm import TFDataset

import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger

from torch.utils.data import DataLoader

import torch
torch.set_float32_matmul_precision('medium')

from safetensors.torch import save_file
from huggingface_hub import HfApi

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
        nb_future=8,
        loss_type="mse",
        parametrization="noisy",
    )

    # logger = TensorBoardLogger("tb_logs/", name="g2pt_grid")
    logger = WandbLogger(project="meteolibre_model_latent")

    trainer = pl.Trainer(
        max_time={"hours": 10},
        logger=logger,
        accumulate_grad_batches=2,
        #fast_dev_run=True,
        #accelerator="cpu", # debug
        gradient_clip_val=1.0,
        log_every_n_steps=5,
        enable_checkpointing=False,
    )  # fast_dev_run=True for quick debugging

    trainer.fit(
        model, train_dataloader, val_dataloader
    )  # Pass val_dataloader if you have validation step in model


    # Save the model in safetensors format
    save_file(model.model.state_dict(), "diffusion_pytorch_model.safetensors")

    #torch.save(model.model.state_dict(), "model_vae.pt")


    # push file to hub
    api = HfApi()
    api.upload_file(
        path_or_fileobj="diffusion_pytorch_model.safetensors",
        path_in_repo="weights_dit3d/diffusion_pytorch_model.safetensors",
        repo_id="Forbu14/meteolibre",
        repo_type="model",
    )


    print("Training finished!")
