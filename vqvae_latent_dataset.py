import os
import torch
from tqdm import tqdm
from torch.utils.data import Dataset

class VQLatentDataset(Dataset):
    def __init__(self, dataloader, vqvae_model, save_path=None, overwrite=False):
        """
        Args:
            dataloader: original image DataLoader
            vqvae_model: trained LitVQVAE (LightningModule)
            save_path: optional .pt file to save/reload latent indices
            overwrite: force recomputation
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.latents = []
        self.vqvae_model = vqvae_model.to(device)
        self.device = device if torch.cuda.is_available() else "cpu"

        if save_path and os.path.exists(save_path) and not overwrite:
            print(f"Loading precomputed latents from {save_path}")
            self.latents = torch.load(save_path)
        else:
            print("Extracting latents from VQ-VAE...")
            self.vqvae_model.eval()
            with torch.no_grad():
                for x, _ in tqdm(dataloader):
                    x = x.to(device)
                    z_e = self.vqvae_model.encoder(x)
                    _, _, indices = self.vqvae_model.vq(z_e)  # (B * H * W)
                    B, _, H, W = z_e.shape
                    indices = indices.view(B, H, W).cpu()
                    self.latents.extend(indices)

            if save_path:
                print(f"Saving latent codes to {save_path}")
                torch.save(self.latents, save_path)

    def __len__(self):
        return len(self.latents)

    def __getitem__(self, idx):
        return self.latents[idx]  # shape: (H, W), dtype: long