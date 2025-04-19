import os
import time
import torch


def one_hot(indices, num_classes):
    """
    Converts class indices to one-hot encoding.
    Args:
        indices: Tensor of shape (B,) with class indices
        num_classes: int, number of classes
    Returns:
        One-hot encoded tensor of shape (B, num_classes)
    """
    return torch.nn.functional.one_hot(indices, num_classes).float()


def update_embedding_ema(embedding, ema_cluster_size, ema_embedding, encoder_output, encodings, decay=0.99, epsilon=1e-5):
    """
    EMA update for vector quantization embeddings.

    Args:
        embedding: nn.Parameter of shape (K, D)
        ema_cluster_size: Tensor of shape (K,)
        ema_embedding: Tensor of shape (K, D)
        encoder_output: Tensor of shape (B, D)
        encodings: Tensor of shape (B, K), one-hot
        decay: float, decay rate for EMA
        epsilon: float, small constant for numerical stability

    Returns:
        Updated ema_cluster_size, ema_embedding (detached)
    """
    K = embedding.shape[0]
    device = encoder_output.device

    # Update cluster size
    N = torch.sum(encodings, dim=0)
    updated_cluster_size = decay * ema_cluster_size + (1 - decay) * N

    n = torch.sum(updated_cluster_size)
    updated_cluster_size = ((updated_cluster_size + epsilon) /
                            (n + K * epsilon) * n)

    # Update embeddings
    dw = torch.matmul(encodings.t(), encoder_output)
    updated_ema_embedding = decay * ema_embedding + (1 - decay) * dw

    # Normalize
    embedding.data.copy_(updated_ema_embedding / updated_cluster_size.unsqueeze(1))

    return updated_cluster_size.detach(), updated_ema_embedding.detach()


class Timer:
    """Simple timer utility."""
    def __init__(self):
        self.start = time.time()

    def reset(self):
        self.start = time.time()

    def elapsed(self):
        return time.time() - self.start


def save_checkpoint(model, optimizer, epoch, path):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }, path)


def load_checkpoint(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch']