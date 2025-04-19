import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

class MaskedConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, mask_type='A', **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)
        self.register_buffer('mask', torch.ones_like(self.weight.data))
        _, _, kH, kW = self.weight.size()

        yc, xc = kH // 2, kW // 2
        self.mask[:, :, yc + 1:] = 0
        self.mask[:, :, yc, xc + (mask_type == 'B'):] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)


class PixelCNN(nn.Module):
    def __init__(self, num_embeddings, input_shape, n_channels=64, kernel_size=7, n_layers=7):
        super().__init__()
        self.input_shape = input_shape  # (H, W)
        self.embedding = nn.Embedding(num_embeddings, 1)

        layers = [MaskedConv2d(1, n_channels, kernel_size, padding=kernel_size//2, mask_type='A')]
        for _ in range(n_layers - 2):
            layers.append(nn.ReLU())
            layers.append(MaskedConv2d(n_channels, n_channels, kernel_size, padding=kernel_size//2, mask_type='B'))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(n_channels, num_embeddings, 1))  # logits over codebook entries

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, H, W) → (B, H, W, E) → (B, E, H, W)
        x = self.embedding(x)                  # (B, H, W, E)
        x = x.permute(0, 3, 1, 2).contiguous() # (B, E, H, W)
        return self.net(x)                     # (B, K, H, W)


class LitPixelCNN(pl.LightningModule):
    def __init__(self, num_embeddings, input_shape, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = PixelCNN(num_embeddings=num_embeddings, input_shape=input_shape)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x = batch.long()  # (B, H, W)
        logits = self(x)
        loss = self.loss_fn(logits, x)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch.long()
        logits = self(x)
        loss = self.loss_fn(logits, x)
        self.log("val/loss", loss, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)