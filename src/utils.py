"""
Helper utilities for the Gradio app (inference only, no training code here).

Three main operations:
  1. preprocess_image  — convert any uploaded PIL image to a model-ready tensor
  2. interpolate       — blend two latent vectors across N steps
  3. tensor_to_numpy   — convert a model output tensor to a displayable numpy array
"""

import numpy as np
import torch
from PIL import Image


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------

def preprocess_image(img, device: torch.device) -> torch.Tensor:
    """
    Converts a PIL Image (any size, any mode) to the format the model expects.

    Steps:
      1. Convert to grayscale ('L' mode) — model trained on single-channel images.
      2. Resize to 28×28 using LANCZOS anti-aliasing.
      3. Convert to float tensor and normalise to [0, 1].
      4. Add batch dimension.

    Input  : PIL.Image  (any H, W, any mode)
    Output : (1, 1, 28, 28) float tensor on `device`
    """
    img = img.convert("L")                    # → single-channel grayscale
    img = img.resize((28, 28), Image.LANCZOS) # → 28×28 pixels

    # Convert to numpy, normalise, add channel + batch dims.
    arr = np.array(img, dtype=np.float32) / 255.0  # (28, 28), values [0, 1]
    tensor = torch.from_numpy(arr)                  # (28, 28)
    tensor = tensor.unsqueeze(0).unsqueeze(0)       # (1, 1, 28, 28)
    return tensor.to(device)


# ---------------------------------------------------------------------------
# Latent space interpolation
# ---------------------------------------------------------------------------

def interpolate(z1: torch.Tensor, z2: torch.Tensor, steps: int = 8) -> torch.Tensor:
    """
    Linearly interpolates between two latent vectors.

    For alpha = 0  the output is z1; for alpha = 1 the output is z2.
    The intermediate frames explore the straight line between the two
    points in latent space, revealing how the model's internal
    representation blends between two garments.

    Input  : z1 (1, 32), z2 (1, 32)
    Output : (steps, 32)  — a batch of interpolated latent vectors
    """
    # linspace gives `steps` evenly-spaced values from 0.0 to 1.0.
    alphas = torch.linspace(0.0, 1.0, steps, device=z1.device)  # (steps,)

    # Broadcast: alphas[:,None] is (steps,1), z1/z2 are (1,32) → (steps,32).
    z_interp = (1.0 - alphas[:, None]) * z1 + alphas[:, None] * z2  # (steps, 32)
    return z_interp


# ---------------------------------------------------------------------------
# Tensor → numpy conversion for display
# ---------------------------------------------------------------------------

def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Converts a single decoded image tensor to a uint8 numpy array suitable
    for Gradio's gr.Image component.

    Input  : (1, 1, 28, 28) or (1, 28, 28) float tensor, values in [0, 1]
    Output : (28, 28) uint8 numpy array, values in [0, 255]
    """
    arr = tensor.squeeze().cpu().detach().numpy()  # (28, 28)
    arr = np.clip(arr, 0.0, 1.0)
    arr = (arr * 255).astype(np.uint8)             # [0, 255]
    return arr
