import torch
import torchvision
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from utils import one_hot, update_embedding_ema


class Encoder(nn.Module):
    def __init__(self, in_channels=3, hidden_channels=256, embedding_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 4, stride=2, padding=1),  # 32x32 -> 16x16
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 4, stride=2, padding=1),  # 16x16 -> 8x8
            nn.ReLU(),
            nn.Conv2d(hidden_channels, embedding_dim, 1)  # output shape: (B, D, 8, 8)
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, out_channels=3, hidden_channels=256, embedding_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(embedding_dim, hidden_channels, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_channels, hidden_channels, 4, stride=2, padding=1),  # 8x8 -> 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_channels, out_channels, 4, stride=2, padding=1)  # 16x16 -> 32x32
        )

    def forward(self, x):
        return self.net(x)


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.beta = commitment_cost

        self.embedding = nn.Parameter(torch.randn(num_embeddings, embedding_dim))

    def forward(self, z):
        # z: (B, D, H, W) -> (BHW, D)
        B, D, H, W = z.shape
        z_flat = z.permute(0, 2, 3, 1).contiguous().view(-1, D)

        # Compute distances
        dist = (z_flat.pow(2).sum(1, keepdim=True)
                - 2 * z_flat @ self.embedding.t()
                + self.embedding.pow(2).sum(1))

        # Nearest encoding
        encoding_indices = torch.argmin(dist, dim=1)
        encodings = one_hot(encoding_indices, self.num_embeddings)

        # Quantized latents
        quantized = encodings @ self.embedding
        quantized = quantized.view(B, H, W, D).permute(0, 3, 1, 2).contiguous()

        # Losses
        commitment_loss = self.beta * F.mse_loss(z.detach(), quantized)
        codebook_loss = F.mse_loss(z, quantized.detach())
        loss = commitment_loss + codebook_loss

        # Straight-through estimator
        quantized = z + (quantized - z).detach()

        return quantized, loss, encoding_indices


class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay=0.99, epsilon=1e-5):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.beta = commitment_cost
        self.decay = decay
        self.epsilon = epsilon

        embed = torch.randn(num_embeddings, embedding_dim)
        self.register_buffer("embedding", embed)
        self.register_buffer("ema_cluster_size", torch.zeros(num_embeddings))
        self.register_buffer("ema_embedding", embed.clone())

    def forward(self, z):
        B, D, H, W = z.shape
        z_flat = z.permute(0, 2, 3, 1).contiguous().view(-1, D)

        # Compute distances
        dist = (z_flat.pow(2).sum(1, keepdim=True)
                - 2 * z_flat @ self.embedding.t()
                + self.embedding.pow(2).sum(1))

        encoding_indices = torch.argmin(dist, dim=1)
        encodings = one_hot(encoding_indices, self.num_embeddings)

        quantized = encodings @ self.embedding
        quantized = quantized.view(B, H, W, D).permute(0, 3, 1, 2).contiguous()

        # EMA update
        self.ema_cluster_size, self.ema_embedding = update_embedding_ema(
            self.embedding, self.ema_cluster_size, self.ema_embedding, z_flat, encodings,
            decay=self.decay, epsilon=self.epsilon
        )

        # Straight-through estimator
        quantized = z + (quantized - z).detach()

        commitment_loss = self.beta * F.mse_loss(z, quantized.detach())
        return quantized, commitment_loss, encoding_indices


class LitVQVAE(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        self.encoder = Encoder(
            in_channels=config["in_channels"],
            hidden_channels=config["hidden_channels"],
            embedding_dim=config["embedding_dim"]
        )
        self.decoder = Decoder(
            out_channels=config["in_channels"],
            hidden_channels=config["hidden_channels"],
            embedding_dim=config["embedding_dim"]
        )
        self.use_ema = config["use_ema"]
        if self.use_ema:
            self.vq = VectorQuantizerEMA(
                num_embeddings=config["num_embeddings"],
                embedding_dim=config["embedding_dim"],
                commitment_cost=config["beta"],
                decay=config["decay"],
                epsilon=config["epsilon"]
            )
        else:
            self.vq = VectorQuantizer(
                num_embeddings=config["num_embeddings"],
                embedding_dim=config["embedding_dim"],
                commitment_cost=config["beta"]
            )

        self.recon_loss_fn = nn.MSELoss()

    def forward(self, x):
        z_e = self.encoder(x)
        z_q, vq_loss, encoding_indices = self.vq(z_e)
        x_recon = self.decoder(z_q)
        return x_recon, vq_loss, encoding_indices

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x_recon, vq_loss, encoding_indices = self.forward(x)
        recon_loss = self.recon_loss_fn(x_recon, x)
        loss = recon_loss + vq_loss
        self.log("train/recon_loss", recon_loss, on_step=False, on_epoch=True)
        self.log("train/vq_loss", vq_loss, on_step=False, on_epoch=True)
        self.log("train/total_loss", loss, on_step=False, on_epoch=True)

        # === Embedding usage and entropy logging ===
        with torch.no_grad():
            probs = torch.bincount(encoding_indices, minlength=self.hparams["num_embeddings"]).float()
            probs /= probs.sum() + 1e-8

            usage = torch.sum(probs > 0).item()
            entropy = -torch.sum(probs * torch.log(probs + 1e-8)).item()

            self.log("train/embedding_usage", usage, on_epoch=True)
            self.log("train/embedding_entropy", entropy, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        x_recon, vq_loss, encoding_indices = self.forward(x)
        # Embedding stats
        with torch.no_grad():
            probs = torch.bincount(encoding_indices, minlength=self.hparams["num_embeddings"]).float()
            probs /= probs.sum() + 1e-8

            usage = torch.sum(probs > 0).item()
            entropy = -torch.sum(probs * torch.log(probs + 1e-8)).item()

            self.log("val/embedding_usage", usage, on_epoch=True)
            self.log("val/embedding_entropy", entropy, on_epoch=True)
        # Losses
        recon_loss = self.recon_loss_fn(x_recon, x)
        loss = recon_loss + vq_loss
        self.log("val/recon_loss", recon_loss, on_step=False, on_epoch=True)
        self.log("val/vq_loss", vq_loss, on_step=False, on_epoch=True)
        self.log("val/total_loss", loss, on_step=False, on_epoch=True)

        # === TensorBoard Image Logging ===
        if batch_idx == 0:  # only log once per epoch
            num_images = min(8, x.size(0))
            x_vis = (x[:num_images] + 1) / 2.0
            x_recon_vis = (x_recon[:num_images] + 1) / 2.0

            grid = torchvision.utils.make_grid(torch.cat([x_vis, x_recon_vis]), nrow=num_images)
            self.logger.experiment.add_image("val/reconstructions", grid, self.current_epoch)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)