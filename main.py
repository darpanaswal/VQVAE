import os
import argparse
from model import LitVQVAE
import pytorch_lightning as pl
from datasets import get_dataset
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

def parse_args():
    base_directory = os.getcwd()
    base_directory += "/VQVAE"
    parser = argparse.ArgumentParser(description="Train VQ-VAE")

    # Training config
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=2e-4)

    # Model config
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--hidden_channels", type=int, default=256)
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--num_embeddings", type=int, default=512)
    parser.add_argument("--beta", type=float, default=0.25)
    parser.add_argument("--dataset", type=str, default="cifar10",
                    choices=["cifar10", "mnist", "fashionmnist"],
                    help="Dataset to train on")
    parser.add_argument("--decay", type=float, default=0.99)
    parser.add_argument("--epsilon", type=float, default=1e-5)
    parser.add_argument("--use_ema", action="store_true", help="Enable EMA for VQ updates")

    # Logging & saving
    parser.add_argument("--log_dir", type=str, default=f"{base_directory}/runs")
    parser.add_argument("--save_dir", type=str, default=f"{base_directory}/checkpoints")

    return parser.parse_args()


def main():
    args = parse_args()

    # Convert args to config dict
    config = vars(args)

    if config["dataset"] in {"mnist", "fashionmnist"}:
        config["in_channels"] = 1

    # Load data
    train_loader, val_loader = get_dataset(name=config["dataset"], batch_size=config["batch_size"])

    # Model
    model = LitVQVAE(config)

    # Logging
    logger = TensorBoardLogger(config["log_dir"], name="vqvae")
    checkpoint_cb = ModelCheckpoint(
        monitor="val/total_loss",
        save_top_k=1,
        mode="min",
        dirpath=config["save_dir"],
        filename="vqvae-{epoch:02d}-{val_total_loss:.4f}"
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=config["epochs"],
        logger=logger,
        callbacks=[checkpoint_cb],
        accelerator="gpu",         # auto-select GPU
        devices="auto",            # use all available GPUs
        strategy="ddp",            # DDP strategy
        precision=16,              # optional: use mixed precision (can be 32 or 16)
        deterministic=True         # for reproducibility
    )

    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
