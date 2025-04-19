import os
import torch
import torchaudio
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.datasets import MNIST, FashionMNIST

base_directory = os.getcwd()
base_directory += "/VQVAE"

def get_cifar10(batch_size=128, num_workers=4, root='./data'):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.CIFAR10(root=root, train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root=root, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader


def get_vctk(batch_size=32, num_workers=4, root=f'./{base_directory}/data/VCTK', sample_rate=16000):
    transform = torchaudio.transforms.Resample(orig_freq=48000, new_freq=sample_rate)

    def collate_fn(batch):
        waveforms = []
        for waveform, _, _, _, _ in batch:
            waveform = transform(waveform)
            waveforms.append(waveform.squeeze(0))  # Remove channel dim if mono
        waveforms = torch.nn.utils.rnn.pad_sequence(waveforms, batch_first=True)
        return waveforms

    dataset = torchaudio.datasets.VCTK(root=root, download=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)

    return loader

def get_mnist(batch_size=128, num_workers=4, root=f"./{base_directory}/data", dataset="mnist"):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset_cls = MNIST if dataset == "mnist" else FashionMNIST

    train_dataset = dataset_cls(root=root, train=True, download=True, transform=transform)
    test_dataset = dataset_cls(root=root, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader


def get_dataset(name="cifar10", batch_size=128, num_workers=4, root=f"./{base_directory}/data"):
    name = name.lower()
    if name == "cifar10":
        return get_cifar10(batch_size, num_workers, root)
    elif name in {"mnist", "fashionmnist"}:
        return get_mnist(batch_size, num_workers, root, dataset=name)
    else:
        raise ValueError(f"Unsupported dataset: {name}")
