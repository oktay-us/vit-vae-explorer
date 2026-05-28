# Project Spec: ViT-VAE Latent Space Explorer

## What We're Building

An end-to-end ML project split across two environments:

| Where | What |
|---|---|
| **Google Colab** (free T4 GPU) | Train the ViT-VAE, save checkpoint to Google Drive |
| **Your PC** (CPU, Docker) | Run the Gradio web app using the downloaded checkpoint |

---

## Dataset: Fashion-MNIST

Downloaded automatically by `torchvision` — no manual setup needed.

- 60,000 training / 10,000 test images
- 28×28 grayscale, 10 clothing categories
- Small enough to train in ~25 min on a T4
- Latent space is genuinely interesting — the model must separate shirts from coats, sandals from sneakers

---

## Model Architecture: ViT-VAE

```
Input: 28×28 grayscale image
  │
  ▼  Patch Embedding  (patch_size=4 → 49 patches, dim=128)
  │
  ▼  Transformer Encoder  (4 layers, 4 heads)
  │
  ▼  CLS token → Linear → μ, log σ²  (latent_dim=32)
  │                └── reparameterization → z ∈ ℝ³²
  ▼  Linear (z → 49×128) → Transformer Decoder  (2 layers)
  │
  ▼  Linear projection → 28×28 reconstruction
```

~3M parameters. Trains in ~25 min on Colab T4, fast for inference on CPU.

---

## Project Structure

```
vit-vae-explorer/
├── notebooks/
│   └── train_colab.ipynb       # everything for Colab: data, model, training
├── checkpoints/
│   └── best_model.pth          # downloaded from Google Drive after training
├── src/
│   ├── model.py                # ViT encoder + VAE — shared between notebook & app
│   ├── dataset.py              # DataModule / transforms
│   └── utils.py                # interpolation, sampling helpers
├── app/
│   └── gradio_app.py           # Gradio web UI (loads checkpoint, runs on CPU)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt            # CPU-only, for Docker
└── README.md
```

> `model.py` is the single source of truth — imported by both the Colab notebook and the Gradio app.
> This means the checkpoint saved on Colab loads perfectly into the local Docker container.

---

## Training (on Google Colab)

### Setup

1. Go to https://colab.research.google.com
2. Runtime → Change runtime type → **T4 GPU**
3. Open `notebooks/train_colab.ipynb`

```python
# Top of notebook — mount Drive to save checkpoints
from google.colab import drive
drive.mount('/content/drive')

!pip install timm einops -q
```

### Hyperparameters

| Parameter | Value | Reason |
|---|---|---|
| Epochs | 30 | Converges well, ~25 min on T4 |
| Batch size | 128 | Fits T4 comfortably |
| Optimizer | AdamW, lr=1e-3 | Standard for transformers |
| β (KL weight) | 0.5 | Smoother latent space (β-VAE) |
| Loss | MSE + β·KL | Good for grayscale reconstruction |

### Saving the checkpoint

```python
import torch
# Save to Google Drive so it survives session end
torch.save(model.state_dict(), '/content/drive/MyDrive/vit_vae_best.pth')
```

---

## Getting the Checkpoint to Your PC

After training:

1. Open Google Drive in your browser
2. Find `vit_vae_best.pth`
3. Download it into `checkpoints/best_model.pth` in your local repo

That's it. No conversion needed — PyTorch checkpoints are portable.

---

## Gradio App (runs locally on CPU)

Three tabs:

**Reconstruct** — Upload or pick a test image. See original vs. reconstruction side-by-side. Latent vector shown as a bar chart.

**Interpolate** — Pick two test images. A slider blends `z = α·z₁ + (1-α)·z₂` and decodes it. Shows 8 interpolation frames as a filmstrip.

**Latent Sampler** — 32 sliders (one per latent dim, range −3 to +3). Move any slider to watch the decoded image update. "Random Sample" button draws z ~ N(0,I).

---

## Docker Deployment (your PC)

### Dockerfile

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 7860
CMD ["python", "app/gradio_app.py"]
```

### requirements.txt (CPU-only)

```
torch --index-url https://download.pytorch.org/whl/cpu
torchvision --index-url https://download.pytorch.org/whl/cpu
timm
einops
gradio
numpy
pillow
```

### Running it

```bash
# Build the image
docker build -t vit-vae-explorer .

# Run on CPU (your PC)
docker run -p 7860:7860 vit-vae-explorer

# Run on GPU (any other machine with nvidia-container-toolkit)
docker run --gpus all -p 7860:7860 vit-vae-explorer
```

Open http://localhost:7860 in your browser.

---

## Milestones

| # | Where | Task | Est. Time |
|---|---|---|---|
| 1 | PC | Repo setup, folder structure, VSCode open | 15 min |
| 2 | PC + Claude Code | Write `src/model.py` (ViT encoder + VAE) | 45 min |
| 3 | PC + Claude Code | Write `notebooks/train_colab.ipynb` | 30 min |
| 4 | Colab | Upload notebook, run training (30 epochs) | 30 min active + 25 min training |
| 5 | PC | Download checkpoint from Drive | 5 min |
| 6 | PC + Claude Code | Write `app/gradio_app.py` (3-tab UI) | 1 hr |
| 7 | PC + Claude Code | Write `Dockerfile` + `requirements.txt` | 20 min |
| 8 | PC | `docker build` + `docker run`, test in browser | 20 min |

**Total active time: ~4 hours** (excluding the 25 min training run)

---

## Device-Agnostic Code Pattern

Every file uses this — write it once, runs anywhere:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ViTVAE().to(device)

# Loading checkpoint — always map to current device
state_dict = torch.load("checkpoints/best_model.pth", map_location=device)
model.load_state_dict(state_dict)
```

No code changes needed between Colab (GPU) and Docker on your PC (CPU).

---

## What You'll Have Learned

- ViT patch embeddings applied to image encoding
- Reparameterization trick in a transformer context
- β-VAE and how KL weight shapes the latent space
- Google Colab as a free GPU training environment
- Portable PyTorch checkpoints (train anywhere, run anywhere)
- Gradio for rapid ML UI prototyping
- Dockerizing a Python ML app (CPU-optimized)
- Claude Code as a coding pair-programmer in a real codebase
