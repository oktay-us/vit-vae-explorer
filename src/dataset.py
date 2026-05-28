"""
Fashion-MNIST data loading.

Fashion-MNIST is a drop-in replacement for the original MNIST digit dataset.
It contains 70,000 grayscale images (28×28) across 10 clothing categories:
  0: T-shirt/top   1: Trouser      2: Pullover    3: Dress      4: Coat
  5: Sandal        6: Shirt        7: Sneaker     8: Bag        9: Ankle boot

torchvision downloads it automatically the first time; ~30 MB total.
"""

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# Human-readable class names in label order.
CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def get_dataloaders(
    data_dir: str = "./data",
    batch_size: int = 128,
    num_workers: int = 2,
):
    """
    Returns train and test DataLoaders for Fashion-MNIST.

    Each item is a tuple (image, label) where:
      image : (1, 28, 28) float tensor  — pixel values in [0, 1]
      label : int                        — class index 0-9

    Args:
        data_dir    : where to store / look for the raw dataset files
        batch_size  : images per batch (128 fits a Colab T4 comfortably)
        num_workers : parallel data-loading workers
                      (set to 0 on Windows if you get multiprocessing errors)
    """
    # ToTensor() converts a PIL Image (H, W) with uint8 values [0,255]
    # to a float tensor (1, H, W) with values in [0.0, 1.0].
    # No extra normalisation needed — the model outputs sigmoid [0,1] too.
    transform = transforms.Compose([
        transforms.ToTensor(),   # (1, 28, 28), values [0, 1]
    ])

    train_dataset = datasets.FashionMNIST(
        root=data_dir, train=True, download=True, transform=transform
    )
    # 60,000 training images

    test_dataset = datasets.FashionMNIST(
        root=data_dir, train=False, download=True, transform=transform
    )
    # 10,000 test images

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,            # shuffle every epoch for better generalisation
        num_workers=num_workers,
        pin_memory=True,         # faster CPU→GPU transfer (no-op on CPU-only machines)
        drop_last=True,          # keep batches uniform size during training
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader
