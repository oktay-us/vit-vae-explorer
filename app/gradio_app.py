"""
Gradio web app for the ViT-VAE Latent Space Explorer.

Three tabs:
  1. Reconstruct   — upload an image and compare it to the model's reconstruction
  2. Interpolate   — blend between two images through latent space
  3. Latent Sampler— control each of the 32 latent dimensions with a slider

Run locally:
  python app/gradio_app.py

Or via Docker:
  docker run -p 7860:7860 vit-vae-explorer

Then open http://localhost:7860 in your browser.
"""

import os
import sys

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

# Allow imports from the project root regardless of where the script is called from.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import ViTVAE
from src.utils import interpolate, preprocess_image, tensor_to_numpy

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

CHECKPOINT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "checkpoints", "best_model.pth",
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ViTVAE()

if os.path.exists(CHECKPOINT):
    state_dict = torch.load(CHECKPOINT, map_location=device)
    model.load_state_dict(state_dict)
    print(f"Loaded checkpoint from {CHECKPOINT}")
else:
    print(
        f"WARNING: checkpoint not found at {CHECKPOINT}\n"
        "The app will run but outputs will be random (untrained model).\n"
        "Train on Colab and download best_model.pth to checkpoints/ first."
    )

model = model.to(device)
model.eval()

# ---------------------------------------------------------------------------
# Tab 1: Reconstruct
# ---------------------------------------------------------------------------

def reconstruct(pil_img):
    """
    Encodes the uploaded image into a 32-d latent vector, then decodes it
    back.  Returns the reconstruction and a bar chart of the latent vector.

    Args:
        pil_img : PIL.Image from the user upload

    Returns:
        recon_img  : (28, 28) uint8 numpy array — reconstructed image
        latent_fig : matplotlib Figure — bar chart of mu (32 values)
    """
    if pil_img is None:
        return None, None

    x = preprocess_image(pil_img, device)   # (1, 1, 28, 28)

    with torch.no_grad():
        mu, log_var = model.encode(x)        # (1, 32), (1, 32)
        z = mu                               # use the mean (no noise) for a deterministic result
        x_recon = model.decode(z)            # (1, 1, 28, 28)

    recon_img = tensor_to_numpy(x_recon)    # (28, 28) uint8

    # Build a bar chart showing the value of each latent dimension.
    mu_np = mu.squeeze().cpu().numpy()      # (32,)
    fig, ax = plt.subplots(figsize=(10, 2))
    colors = ["steelblue" if v >= 0 else "tomato" for v in mu_np]
    ax.bar(range(32), mu_np, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Latent dimension")
    ax.set_ylabel("μ value")
    ax.set_title("Latent vector (μ)")
    ax.set_xlim(-0.5, 31.5)
    fig.tight_layout()

    return recon_img, fig


# ---------------------------------------------------------------------------
# Tab 2: Interpolate
# ---------------------------------------------------------------------------

def interpolate_images(pil_img1, pil_img2):
    """
    Encodes both images, then decodes 8 evenly-spaced blends along the
    straight line between their latent vectors.

    Args:
        pil_img1, pil_img2 : PIL.Image uploads

    Returns:
        List of 8 (28, 28) uint8 numpy arrays — the interpolation filmstrip
    """
    if pil_img1 is None or pil_img2 is None:
        return []

    x1 = preprocess_image(pil_img1, device)  # (1, 1, 28, 28)
    x2 = preprocess_image(pil_img2, device)  # (1, 1, 28, 28)

    with torch.no_grad():
        mu1, _ = model.encode(x1)            # (1, 32)
        mu2, _ = model.encode(x2)            # (1, 32)

        z_interp = interpolate(mu1, mu2, steps=8)  # (8, 32)
        frames = model.decode(z_interp)             # (8, 1, 28, 28)

    # Convert each frame to a displayable image.
    frame_list = [tensor_to_numpy(frames[i]) for i in range(8)]  # list of (28,28) uint8
    return frame_list


# ---------------------------------------------------------------------------
# Tab 3: Latent Sampler
# ---------------------------------------------------------------------------

def decode_from_sliders(*slider_vals):
    """
    Decodes a latent vector constructed from the 32 slider values.
    Called every time any slider moves.

    Args:
        *slider_vals : 32 float values (one per slider)

    Returns:
        (28, 28) uint8 numpy array — decoded image
    """
    # Collect all 32 slider values into a tensor.
    z = torch.tensor(list(slider_vals), dtype=torch.float32)  # (32,)
    z = z.unsqueeze(0).to(device)                              # (1, 32)

    with torch.no_grad():
        x_recon = model.decode(z)   # (1, 1, 28, 28)

    return tensor_to_numpy(x_recon)  # (28, 28) uint8


def random_sample_values():
    """
    Draws a random latent vector from the prior N(0, I) and returns
    32 individual values to update all sliders simultaneously.
    """
    z = torch.randn(32)                         # (32,)
    return [float(z[i].item()) for i in range(32)]


# ---------------------------------------------------------------------------
# Build the Gradio interface
# ---------------------------------------------------------------------------

with gr.Blocks(title="ViT-VAE Latent Space Explorer") as demo:
    gr.Markdown("# ViT-VAE Latent Space Explorer")
    gr.Markdown(
        "A Vision Transformer VAE trained on Fashion-MNIST. "
        "Upload clothing images to reconstruct them, interpolate between two garments, "
        "or hand-craft a latent vector with the sliders below."
    )

    # ── Tab 1: Reconstruct ──────────────────────────────────────────────────
    with gr.Tab("Reconstruct"):
        gr.Markdown(
            "Upload any grayscale image of a garment. The app converts it to 28×28, "
            "encodes it to a 32-d latent vector, then decodes it back."
        )
        with gr.Row():
            with gr.Column():
                recon_input = gr.Image(type="pil", label="Upload image")
                recon_btn = gr.Button("Reconstruct", variant="primary")
            with gr.Column():
                recon_output = gr.Image(label="Reconstruction", image_mode="L")
        latent_chart = gr.Plot(label="Latent vector (μ)")

        recon_btn.click(
            fn=reconstruct,
            inputs=recon_input,
            outputs=[recon_output, latent_chart],
        )

    # ── Tab 2: Interpolate ──────────────────────────────────────────────────
    with gr.Tab("Interpolate"):
        gr.Markdown(
            "Upload two garment images. The app encodes both to latent vectors "
            "and shows 8 frames blending from one to the other through latent space."
        )
        with gr.Row():
            interp_img1 = gr.Image(type="pil", label="Image A")
            interp_img2 = gr.Image(type="pil", label="Image B")
        interp_btn = gr.Button("Interpolate", variant="primary")
        interp_gallery = gr.Gallery(
            label="Interpolation (A → B)",
            columns=8,
            rows=1,
            height=160,
        )

        interp_btn.click(
            fn=interpolate_images,
            inputs=[interp_img1, interp_img2],
            outputs=interp_gallery,
        )

    # ── Tab 3: Latent Sampler ───────────────────────────────────────────────
    with gr.Tab("Latent Sampler"):
        gr.Markdown(
            "Each slider controls one of the 32 latent dimensions (range −3 to +3). "
            "Move any slider to watch the decoded image update in real time. "
            "Click **Random Sample** to draw a random point from the prior N(0, I)."
        )
        sampler_output = gr.Image(label="Decoded image", image_mode="L")
        random_btn = gr.Button("Random Sample")

        # Build 32 sliders programmatically; 4 per row for a compact layout.
        sliders = []
        for row_start in range(0, 32, 4):
            with gr.Row():
                for i in range(row_start, min(row_start + 4, 32)):
                    s = gr.Slider(
                        minimum=-3.0,
                        maximum=3.0,
                        value=0.0,
                        step=0.05,
                        label=f"z[{i}]",
                    )
                    sliders.append(s)

        # Any slider change re-runs the decode.
        for slider in sliders:
            slider.change(
                fn=decode_from_sliders,
                inputs=sliders,
                outputs=sampler_output,
            )

        # Random Sample resets all sliders, which triggers the decode chain.
        random_btn.click(
            fn=random_sample_values,
            outputs=sliders,
        )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
