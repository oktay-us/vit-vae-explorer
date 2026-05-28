"""
Latent-space classifier for Fashion-MNIST.

Instead of classifying raw pixels, we classify the 32-d latent vector (mu)
produced by the frozen VAE encoder.  This is much easier than pixel classification
because the VAE has already compressed the image into a structured representation.

Architecture:
  z (B, 32) → Linear(32,128) → ReLU → Dropout → Linear(128,64) → ReLU → Linear(64,10)
  Output: (B, 10) logits

Run this file directly to train and save the classifier:
  python src/classifier.py
"""

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import ViTVAE

CLASSIFIER_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "checkpoints", "classifier.pth",
)

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class LatentClassifier(nn.Module):
    """
    Small MLP that maps a 32-d latent vector to one of 10 clothing classes.

    Input  : (B, 32)  — mu from the VAE encoder (or slider z values)
    Output : (B, 10)  — raw logits; apply softmax for probabilities
    """
    def __init__(self, latent_dim: int = 32, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)   # (B, 10) logits


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_classifier(
    vae_checkpoint: str = None,
    data_dir: str = "./data",
    epochs: int = 20,
    batch_size: int = 256,
    lr: float = 1e-3,
    save_path: str = None,
    device: torch.device = None,
):
    """
    Trains the latent classifier by:
      1. Loading the frozen VAE encoder.
      2. Encoding all Fashion-MNIST images to 32-d mu vectors.
      3. Training the MLP classifier on those (mu, label) pairs.

    The VAE weights are frozen — only the classifier is trained.

    Args:
        vae_checkpoint : path to best_model.pth (auto-detected if None)
        data_dir       : Fashion-MNIST data directory
        epochs         : training epochs (20 is plenty for this simple task)
        batch_size     : images per batch
        lr             : Adam learning rate
        save_path      : where to save classifier.pth (auto-detected if None)
        device         : auto-detected if None
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training classifier on: {device}")

    if vae_checkpoint is None:
        vae_checkpoint = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "checkpoints", "best_model.pth",
        )
    if save_path is None:
        save_path = CLASSIFIER_PATH

    # Load the frozen VAE encoder — weights never change during classifier training.
    vae = ViTVAE().to(device)
    state_dict = torch.load(vae_checkpoint, map_location=device)
    vae.load_state_dict(state_dict)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False   # freeze all VAE weights

    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.FashionMNIST(data_dir, train=True,  download=True, transform=transform)
    test_ds  = datasets.FashionMNIST(data_dir, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=2)

    classifier = LatentClassifier().to(device)
    optimizer  = torch.optim.Adam(classifier.parameters(), lr=lr)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0

    for epoch in range(1, epochs + 1):

        # ── Train ────────────────────────────────────────────────────────────
        classifier.train()
        total_loss = correct = total = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            # imgs: (B, 1, 28, 28), labels: (B,)

            with torch.no_grad():
                mu, _ = vae.encode(imgs)   # (B, 32) — frozen encoder, no grad

            logits = classifier(mu)        # (B, 10)
            loss   = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct    += (logits.argmax(1) == labels).sum().item()
            total      += labels.size(0)

        train_acc = correct / total * 100

        # ── Validate ─────────────────────────────────────────────────────────
        classifier.eval()
        correct = total = 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                mu, _ = vae.encode(imgs)
                logits = classifier(mu)
                correct += (logits.argmax(1) == labels).sum().item()
                total   += labels.size(0)

        val_acc = correct / total * 100
        scheduler.step()

        print(f"Epoch {epoch:2d}/{epochs}  train_acc={train_acc:.1f}%  val_acc={val_acc:.1f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(classifier.state_dict(), save_path)
            print(f"  >> Saved best classifier -> {save_path}  (val_acc={val_acc:.1f}%)")

    print(f"\nDone. Best validation accuracy: {best_acc:.1f}%")
    return classifier


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train_classifier()
