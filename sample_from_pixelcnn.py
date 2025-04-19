import torch
import argparse
import torchvision
from model import LitVQVAE
import matplotlib.pyplot as plt
from pixelcnn_model import LitPixelCNN


@torch.no_grad()
def sample_latents(pixelcnn, shape, device):
    pixelcnn.eval()
    H, W = shape
    num_embeddings = pixelcnn.hparams.num_embeddings
    samples = torch.zeros((1, H, W), dtype=torch.long, device=device)

    for i in range(H):
        for j in range(W):
            logits = pixelcnn(samples)
            probs = torch.softmax(logits[0, :, i, j], dim=0)
            samples[0, i, j] = torch.multinomial(probs, 1)

    return samples  # shape: (1, H, W)


@torch.no_grad()
def decode_latents(vqvae, latent_indices):
    vqvae.eval()
    embedding = vqvae.vq.embedding

    z_q = embedding[latent_indices.view(-1)]    # (H * W, D)
    H, W = latent_indices.shape[-2:]
    z_q = z_q.view(1, H, W, -1).permute(0, 3, 1, 2).contiguous()  # (1, D, H, W)

    return vqvae.decoder(z_q)  # shape: (1, C, H', W')


def visualize(img_tensor, save_path=None):
    img = img_tensor.squeeze(0).cpu()
    img = (img + 1) / 2  # [-1,1] to [0,1]
    grid = torchvision.utils.make_grid(img, nrow=1)
    plt.figure(figsize=(4, 4))
    plt.axis("off")
    plt.imshow(grid.permute(1, 2, 0))
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="Sample images from PixelCNN + VQ-VAE decoder")
    parser.add_argument("--vqvae_ckpt", type=str, required=True)
    parser.add_argument("--pixelcnn_ckpt", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load models
    vqvae = LitVQVAE.load_from_checkpoint(args.vqvae_ckpt)
    pixelcnn = LitPixelCNN.load_from_checkpoint(args.pixelcnn_ckpt, 
                                                num_embeddings=vqvae.hparams["num_embeddings"],
                                                input_shape=(8, 8))  # Update shape as needed

    vqvae.to(device)
    pixelcnn.to(device)

    # Sample latent codes
    latent_indices = sample_latents(pixelcnn, shape=(8, 8), device=device)  # (1, 8, 8)

    # Decode into image
    img = decode_latents(vqvae, latent_indices[0])  # remove batch dim from latent_indices

    # Show or save
    visualize(img, save_path=args.output)


if __name__ == "__main__":
    main()