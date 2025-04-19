import os
import torch
import argparse
from model import LitVQVAE
import pytorch_lightning as pl
from datasets import get_dataset
from pixelcnn_model import LitPixelCNN
from torch.utils.data import DataLoader
from vqvae_latent_dataset import VQLatentDataset
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger


def parse_args():
    base_directory = os.getcwd()
    base_directory += "/VQVAE"
    parser = argparse.ArgumentParser(description="Train PixelCNN Prior on VQ-VAE Latents")

    parser.add_argument("--vqvae_ckpt", type=str, required=True, help="Path to trained VQ-VAE checkpoint")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "mnist", "fashionmnist"])
    parser.add_argument("--latent_cache", type=str, default="latents.pt")
    parser.add_argument("--overwrite_latents", action="store_true")

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument("--log_dir", type=str, default=f"{base_directory}/runs/pixelcnn")
    parser.add_argument("--save_dir", type=str, default=f"{base_directory}/checkpoints/pixelcnn")

    return parser.parse_args()


def main():
    args = parse_args()

    # === Load original data + VQ-VAE ===
    _, test_loader = get_dataset(name=args.dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    vqvae = LitVQVAE.load_from_checkpoint(args.vqvae_ckpt)
    vqvae.eval()

    # === Generate or load latent dataset ===
    latent_dataset = VQLatentDataset(
        dataloader=test_loader,
        vqvae_model=vqvae,
        save_path=args.latent_cache,
        overwrite=args.overwrite_latents
    )
    latent_loader = DataLoader(latent_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # === Get latent shape & codebook size from VQ-VAE ===
    example_latent = latent_dataset[0]
    input_shape = example_latent.shape  # e.g. (8, 8)
    num_embeddings = vqvae.hparams["num_embeddings"]

    # === Init PixelCNN ===
    model = LitPixelCNN(num_embeddings=num_embeddings, input_shape=input_shape, lr=args.lr)

    # === Logger & Checkpointing ===
    logger = TensorBoardLogger(save_dir=args.log_dir, name=args.dataset)
    checkpoint_cb = ModelCheckpoint(
        monitor="val/loss",
        save_top_k=1,
        mode="min",
        dirpath=args.save_dir,
        filename="pixelcnn-{epoch:02d}-{val_loss:.4f}"
    )

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        logger=logger,
        callbacks=[checkpoint_cb],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices="auto",
        strategy="ddp" if torch.cuda.device_count() > 1 else "auto"
    )

    trainer.fit(model, latent_loader, val_dataloaders=latent_loader)


if __name__ == "__main__":
    main()