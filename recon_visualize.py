import torch
import argparse
from model import LitVQVAE
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from datasets import get_dataset


def show_reconstructions(model, dataloader, device, num_samples=8, save_path=None):
    model.eval()
    model.to(device)

    with torch.no_grad():
        x, _ = next(iter(dataloader))
        x = x[:num_samples].to(device)
        x_recon, _ = model(x)

        x = x.cpu().numpy().transpose(0, 2, 3, 1)
        x_recon = x_recon.cpu().numpy().transpose(0, 2, 3, 1)

        fig, axs = plt.subplots(2, num_samples, figsize=(num_samples * 2, 4))
        for i in range(num_samples):
            axs[0, i].imshow((x[i] + 1) / 2)  # from [-1,1] back to [0,1]
            axs[0, i].axis("off")
            axs[0, i].set_title("Original")

            axs[1, i].imshow((x_recon[i] + 1) / 2)
            axs[1, i].axis("off")
            axs[1, i].set_title("Reconstructed")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize VQ-VAE Reconstructions")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "mnist", "fashionmnist"])
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--save_path", type=str, default=None, help="Optional path to save the plot")
    return parser.parse_args()


def main():
    args = parse_args()
    config = torch.load(args.checkpoint)["hyper_parameters"]

    # Dataset-aware channel override
    if args.dataset in {"mnist", "fashionmnist"}:
        config["in_channels"] = 1

    _, test_loader = get_dataset(name=args.dataset, batch_size=32)

    model = LitVQVAE(config)
    model.load_from_checkpoint(args.checkpoint, config=config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    show_reconstructions(model, test_loader, device, num_samples=args.num_samples, save_path=args.save_path)


if __name__ == "__main__":
    main()