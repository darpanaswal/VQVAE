import torch
from model import LitVQVAE

config = {
    "in_channels": 1,
    "hidden_channels": 64,
    "embedding_dim": 32,
    "num_embeddings": 128,
    "beta": 0.25,
    "decay": 0.99,
    "epsilon": 1e-5,
    "use_ema": False,
    "lr": 2e-4
}

model = LitVQVAE(config)
dummy_input = torch.randn(8, 1, 28, 28)  # like MNIST
x_recon, vq_loss, indices = model(dummy_input)

print("Reconstructed shape:", x_recon.shape)
print("VQ Loss:", vq_loss.item())
print("Encoding indices shape:", indices.shape)