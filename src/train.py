"""
Training loop for the ViT-VAE.

Loss function — β-VAE:
  total = reconstruction_loss + β * KL_divergence

  reconstruction_loss:
    MSE between the input image and its reconstruction, summed over all
    pixels then averaged over the batch.
    Shape: scalar

  KL_divergence:
    Measures how far the learned posterior q(z|x) = N(mu, sigma^2) is
    from the prior p(z) = N(0, I).
    Closed-form: -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
    Shape: scalar (summed over latent dims, averaged over batch)

  β (default 0.5):
    Weights the KL term.  β < 1 relaxes the constraint, letting the model
    prioritise reconstruction quality.  β > 1 (β-VAE) forces a more
    disentangled latent space.
"""

import os
import torch
import torch.nn.functional as F

from src.model import ViTVAE
from src.dataset import get_dataloaders


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def vae_loss(
    x_recon: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    log_var: torch.Tensor,
    beta: float = 0.5,
):
    """
    Computes the combined VAE loss.

    Args:
        x_recon  : reconstructed images  (B, 1, 28, 28)
        x        : original images       (B, 1, 28, 28)
        mu       : encoder mean          (B, 32)
        log_var  : encoder log-variance  (B, 32)
        beta     : KL weight

    Returns:
        total    : scalar — total loss (backprop target)
        recon    : scalar — reconstruction component (for logging)
        kl       : scalar — KL component (for logging)
    """
    # Sum MSE over all pixels, then average over the batch.
    # reduction='sum' then /B gives a per-image average, which scales better
    # across different batch sizes than reduction='mean' (which divides by B*C*H*W).
    B = x.shape[0]
    recon = F.mse_loss(x_recon, x, reduction="sum") / B   # scalar

    # KL divergence between N(mu, exp(log_var)) and N(0, I).
    # Formula: -0.5 * Σ_d (1 + log_var_d - mu_d^2 - exp(log_var_d))
    kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / B  # scalar

    total = recon + beta * kl
    return total, recon, kl


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    model: ViTVAE = None,
    data_dir: str = "./data",
    epochs: int = 30,
    batch_size: int = 128,
    lr: float = 1e-3,
    beta: float = 0.5,
    save_path: str = "checkpoints/best_model.pth",
    num_workers: int = 2,
    device: torch.device = None,
):
    """
    Trains the ViT-VAE and saves the best checkpoint (lowest validation loss).

    Args:
        model       : ViTVAE instance (created fresh if None)
        data_dir    : path for Fashion-MNIST download / cache
        epochs      : number of full passes over the training set
        batch_size  : images per mini-batch
        lr          : AdamW learning rate
        beta        : KL weight in the VAE loss
        save_path   : where to write the best checkpoint (.pth)
        num_workers : DataLoader worker processes
        device      : torch.device; auto-detected if None

    Returns:
        model       : trained ViTVAE (on the same device)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    if model is None:
        model = ViTVAE()

    model = model.to(device)

    train_loader, test_loader = get_dataloaders(data_dir, batch_size, num_workers)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Cosine annealing reduces the learning rate smoothly to near-zero by the
    # last epoch, which helps avoid oscillating around a minimum.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Create the checkpoint directory if it doesn't exist yet.
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):

        # ── Training phase ──────────────────────────────────────────────────
        model.train()
        train_total = train_recon = train_kl = 0.0

        for x, _ in train_loader:
            # x: (B, 1, 28, 28) — labels ignored, VAE is unsupervised
            x = x.to(device)

            x_recon, mu, log_var = model(x)
            # x_recon: (B, 1, 28, 28), mu/log_var: (B, 32)

            loss, recon, kl = vae_loss(x_recon, x, mu, log_var, beta)

            optimizer.zero_grad()
            loss.backward()
            # Clip gradients to prevent exploding gradients in early training.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_total += loss.item()
            train_recon += recon.item()
            train_kl += kl.item()

        n = len(train_loader)
        train_total /= n
        train_recon /= n
        train_kl /= n

        # ── Validation phase ─────────────────────────────────────────────────
        model.eval()
        val_total = val_recon = val_kl = 0.0

        with torch.no_grad():
            for x, _ in test_loader:
                x = x.to(device)
                x_recon, mu, log_var = model(x)
                loss, recon, kl = vae_loss(x_recon, x, mu, log_var, beta)
                val_total += loss.item()
                val_recon += recon.item()
                val_kl += kl.item()

        n = len(test_loader)
        val_total /= n
        val_recon /= n
        val_kl /= n

        scheduler.step()

        print(
            f"Epoch {epoch:3d}/{epochs}  "
            f"train loss={train_total:.4f} (recon={train_recon:.4f}, kl={train_kl:.4f})  "
            f"val loss={val_total:.4f} (recon={val_recon:.4f}, kl={val_kl:.4f})"
        )

        # Save checkpoint whenever validation loss improves.
        if val_total < best_val_loss:
            best_val_loss = val_total
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ Saved best model → {save_path}  (val_loss={val_total:.4f})")

    return model
