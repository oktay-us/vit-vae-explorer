"""
ViT-VAE: Vision Transformer Variational Autoencoder
Trained on Fashion-MNIST (28x28 grayscale images, 10 clothing classes).

Full dimension flow (B = batch size):
  Input            : (B, 1, 28, 28)
  PatchEmbedding   : (B, 49, 128)   — 49 non-overlapping 4×4 patches, each projected to dim 128
  + CLS token      : (B, 50, 128)   — one learnable classification token prepended
  + pos embedding  : (B, 50, 128)   — sinusoidal-style positional bias added
  Encoder (4L, 4H) : (B, 50, 128)   — 4 Transformer layers with 4 attention heads
  CLS extract      : (B, 128)        — only the CLS token is used as the image summary
  mu / log_var     : (B, 32) each    — two linear heads produce VAE parameters
  z (reparam)      : (B, 32)         — sampled latent vector via reparameterisation trick
  Linear expand    : (B, 49×128)     — project z back up to patch-sequence length
  Decoder (2L, 4H) : (B, 49, 128)   — 2 lighter Transformer layers reconstruct patch tokens
  Patch projection : (B, 49, 16)    — each token → 16 pixel values (4×4 patch)
  Reassemble       : (B, 1, 28, 28) — stitch patches back into the image grid
  Sigmoid          : (B, 1, 28, 28) — squash pixel values to [0, 1]
"""

import warnings

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# 1. Patch Embedding
# ---------------------------------------------------------------------------

class PatchEmbedding(nn.Module):
    """
    Splits a 28×28 image into 4×4 non-overlapping patches and linearly
    projects each patch into a vector of size embed_dim.

    Uses a Conv2d with kernel_size == stride == patch_size so that every
    kernel application touches exactly one patch — equivalent to extracting
    each patch and multiplying by a weight matrix, but faster.

    Input  shape: (B, 1, 28, 28)
    Output shape: (B, 49, 128)
      — 49 = (28/4)^2 = 7×7 patches
      — 128 = embed_dim
    """

    def __init__(
        self,
        img_size: int = 28,
        patch_size: int = 4,
        in_channels: int = 1,
        embed_dim: int = 128,
    ):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2  # 49

        # Conv2d acts as a learned linear projection over each patch.
        # kernel_size=patch_size, stride=patch_size → no overlap between patches.
        # Output channels = embed_dim.
        self.projection = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )
        # After Conv2d: (B, embed_dim=128, 7, 7)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, 28, 28)
        x = self.projection(x)   # → (B, 128, 7, 7)
        x = x.flatten(2)         # → (B, 128, 49)  — flatten the 7×7 spatial grid
        x = x.transpose(1, 2)    # → (B, 49, 128)  — seq_len first for Transformer
        return x


# ---------------------------------------------------------------------------
# 2. Encoder
# ---------------------------------------------------------------------------

class ViTEncoder(nn.Module):
    """
    Transformer encoder that maps a patch sequence to VAE parameters (mu, log_var).

    A learnable CLS token is prepended to the patch sequence so that the
    Transformer can accumulate a global image summary into that single token.
    After encoding, only the CLS token is extracted and passed through two
    linear heads that produce mu and log_var.

    Input  shape: (B, 49, 128)   — patch embeddings from PatchEmbedding
    Output shapes:
      mu      : (B, 32)
      log_var : (B, 32)
    """

    def __init__(
        self,
        num_patches: int = 49,
        embed_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        latent_dim: int = 32,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()

        # CLS token: a single learnable vector added at position 0.
        # Shape (1, 1, embed_dim) → expanded to (B, 1, embed_dim) at runtime.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Positional embedding: one vector per token (49 patches + 1 CLS).
        # Added element-wise so the model knows where each patch came from.
        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_patches + 1, embed_dim) * 0.02
        )

        # Standard Pre-LN Transformer encoder.
        # dim_feedforward = embed_dim * mlp_ratio = 128 * 4 = 512.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            batch_first=True,   # expects (B, seq, dim) — matches our layout
            norm_first=True,    # Pre-LN: more stable training
        )
        # norm_first=True disables a minor nested-tensor optimisation; suppress the noise.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # VAE heads: two independent linear projections from the CLS token.
        self.mu_head = nn.Linear(embed_dim, latent_dim)       # → (B, 32)
        self.log_var_head = nn.Linear(embed_dim, latent_dim)  # → (B, 32)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor):
        # x: (B, 49, 128)
        B = x.shape[0]

        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, 128)
        x = torch.cat([cls, x], dim=1)           # (B, 50, 128)  — CLS first
        x = x + self.pos_embedding               # (B, 50, 128)  — add positional info

        x = self.transformer(x)                  # (B, 50, 128)  — self-attention across all tokens

        cls_out = x[:, 0]                        # (B, 128)       — CLS token holds global summary

        mu = self.mu_head(cls_out)               # (B, 32)
        log_var = self.log_var_head(cls_out)     # (B, 32)
        return mu, log_var


# ---------------------------------------------------------------------------
# 3. Reparameterisation Trick
# ---------------------------------------------------------------------------

def reparameterize(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    """
    Samples z from N(mu, sigma^2) using the reparameterisation trick so that
    gradients flow through mu and log_var during backpropagation.

    z = mu + eps * exp(0.5 * log_var)   where eps ~ N(0, I)

    Input  shapes: (B, 32), (B, 32)
    Output shape : (B, 32)
    """
    # exp(0.5 * log_var) converts log-variance to standard deviation.
    std = torch.exp(0.5 * log_var)   # (B, 32)
    eps = torch.randn_like(std)       # (B, 32)  — same device/dtype automatically
    return mu + eps * std             # (B, 32)


# ---------------------------------------------------------------------------
# 4. Decoder
# ---------------------------------------------------------------------------

class ViTDecoder(nn.Module):
    """
    Transformer decoder that maps a latent vector z back to a 28×28 image.

    The latent vector is first expanded into a sequence of 49 patch tokens
    via a linear layer, then refined by 2 Transformer layers.  A final linear
    head projects each token back to 16 pixel values (4×4 patch), and the
    patches are reassembled into the full image grid.

    Input  shape: (B, 32)        — latent vector z
    Output shape: (B, 1, 28, 28) — reconstructed image, values in [0, 1]
    """

    def __init__(
        self,
        num_patches: int = 49,
        embed_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        latent_dim: int = 32,
        patch_size: int = 4,
        in_channels: int = 1,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_patches = num_patches              # 49
        self.embed_dim = embed_dim                  # 128
        self.patch_size = patch_size                # 4
        self.in_channels = in_channels              # 1
        self.patches_per_side = int(num_patches ** 0.5)  # 7

        # Expand the latent vector into a full patch-token sequence.
        # 32 → 49*128 = 6,272 values, then reshape to (B, 49, 128).
        self.expand = nn.Linear(latent_dim, num_patches * embed_dim)

        # Separate positional embedding for the decoder (not shared with encoder).
        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_patches, embed_dim) * 0.02
        )

        # Lighter Transformer (2 layers vs. encoder's 4) for reconstruction.
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.transformer = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)

        # Project each patch token to pixel values.
        # patch_size^2 * in_channels = 4*4*1 = 16 pixel values per patch.
        self.patch_proj = nn.Linear(embed_dim, patch_size * patch_size * in_channels)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B, 32)
        B = z.shape[0]
        p = self.patch_size          # 4
        h = w = self.patches_per_side  # 7
        C = self.in_channels         # 1

        x = self.expand(z)                          # (B, 49×128 = 6,272)
        x = x.view(B, self.num_patches, self.embed_dim)  # (B, 49, 128)
        x = x + self.pos_embedding                  # (B, 49, 128)

        x = self.transformer(x)                     # (B, 49, 128)

        x = self.patch_proj(x)                      # (B, 49, 16)  — 16 = p*p*C

        # Reassemble patches into the image grid.
        # Each of the 49 tokens represents one 4×4 patch; we place them on a 7×7 grid.
        x = x.view(B, h, w, p, p)                  # (B, 7, 7, 4, 4)
        x = x.permute(0, 1, 3, 2, 4)              # (B, 7, 4, 7, 4)  — interleave rows/cols
        x = x.contiguous().view(B, C, h * p, w * p)  # (B, 1, 28, 28)

        return torch.sigmoid(x)                     # (B, 1, 28, 28)  — squash to [0, 1]


# ---------------------------------------------------------------------------
# 5. Full ViT-VAE
# ---------------------------------------------------------------------------

class ViTVAE(nn.Module):
    """
    Full Vision Transformer VAE.

    Wraps PatchEmbedding → ViTEncoder → reparameterize → ViTDecoder into
    a single module.  The three public methods encode/decode/forward are the
    main entry points used by the training loop and the Gradio app.

    Architecture summary (B = batch size):
      forward()  : (B,1,28,28) → (x_recon, mu, log_var)
      encode()   : (B,1,28,28) → (mu, log_var)  both (B, 32)
      decode()   : (B, 32)     → (B,1,28,28)
    """

    def __init__(
        self,
        img_size: int = 28,
        patch_size: int = 4,
        in_channels: int = 1,
        embed_dim: int = 128,
        enc_layers: int = 4,
        dec_layers: int = 2,
        num_heads: int = 4,
        latent_dim: int = 32,
    ):
        super().__init__()
        num_patches = (img_size // patch_size) ** 2  # 49

        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.encoder = ViTEncoder(num_patches, embed_dim, enc_layers, num_heads, latent_dim)
        self.decoder = ViTDecoder(num_patches, embed_dim, dec_layers, num_heads, latent_dim, patch_size, in_channels)

    def encode(self, x: torch.Tensor):
        """
        Encodes an image to VAE parameters.
        Input  : (B, 1, 28, 28)
        Returns: mu (B, 32), log_var (B, 32)
        """
        patches = self.patch_embed(x)       # (B, 49, 128)
        return self.encoder(patches)        # mu (B,32), log_var (B,32)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decodes a latent vector to a reconstructed image.
        Input  : (B, 32)
        Returns: (B, 1, 28, 28)
        """
        return self.decoder(z)              # (B, 1, 28, 28)

    def forward(self, x: torch.Tensor):
        """
        Full forward pass: encode → sample → decode.
        Input  : (B, 1, 28, 28)
        Returns: x_recon (B,1,28,28), mu (B,32), log_var (B,32)
        """
        mu, log_var = self.encode(x)        # (B,32), (B,32)
        z = reparameterize(mu, log_var)     # (B,32)
        x_recon = self.decode(z)            # (B,1,28,28)
        return x_recon, mu, log_var
