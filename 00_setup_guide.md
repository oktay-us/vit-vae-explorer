# ViT-VAE Project — Setup Guide

## Workflow Overview

```
Google Colab (free GPU)          Your Windows 11 PC
─────────────────────            ─────────────────────────────
Train ViT-VAE                    Git + VSCode + Claude Code
Save checkpoint → Google Drive   Docker Desktop
                                 Download checkpoint → run app
```

**No CUDA install needed on your PC.** Colab handles all the GPU training.
Your machine only runs the Gradio app (inference), which is fast on CPU.

---

## What You Need Locally (Windows 11)

### 1. Git

Download: https://git-scm.com/download/win — accept all defaults.

```powershell
git --version   # verify
```

---

### 2. Python 3.11

Download: https://www.python.org/downloads/release/python-3119/  
**During install:** ✅ Check "Add python.exe to PATH"

```powershell
python --version   # Python 3.11.x
pip --version
```

> No CUDA, no GPU packages needed locally. Plain Python is enough.

---

### 3. Docker Desktop

Download: https://www.docker.com/products/docker-desktop/

During install:
- ✅ "Use WSL 2 instead of Hyper-V" (better performance)
- Open Docker Desktop after install and let it finish initializing

```powershell
docker --version
docker run hello-world   # should print "Hello from Docker!"
```

---

### 4. VSCode

Download: https://code.visualstudio.com/

Install these extensions (Ctrl+Shift+X):

| Extension | Why |
|---|---|
| **Python** (Microsoft) | IntelliSense, linting, run configs |
| **Pylance** (Microsoft) | Fast type checking |
| **Ruff** | Fast formatter/linter |
| **Docker** (Microsoft) | Dockerfile syntax, container management |
| **GitLens** | Inline git blame, history |
| **Jupyter** | Open .ipynb notebooks locally if needed |

---

### 5. Claude Code

First install Node.js (LTS): https://nodejs.org/en

Then:

```powershell
npm install -g @anthropic-ai/claude-code
claude --version   # verify
claude             # first run — follow browser login prompt
```

> In your project folder, just type `claude` to start a session.
> Claude Code reads your files, runs code, edits, and debugs inline.

---

### 6. Local Virtual Environment

```powershell
cd C:\Users\agcao\Desktop\ViT_Claude\ViT_Claude
python -m venv .venv
.venv\Scripts\activate        # (.venv) appears in prompt

# CPU-only packages for running the Gradio app locally
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install gradio einops timm numpy pillow
```

> This installs the CPU-only PyTorch build — small, fast to download, enough for inference.

---

## What You Need on Google Colab

Nothing to install — Colab already has PyTorch + CUDA pre-installed.

You just need:
- A **Google account** (for Drive to save your checkpoint)
- Runtime → Change runtime type → **T4 GPU** (free tier)

Extra packages to install at the top of your Colab notebook:

```python
!pip install timm einops gradio -q
```

---

## Sanity Checks

Run these on your PC before starting:

```powershell
git --version
python --version
docker --version
claude --version
python -c "import torch; print(torch.__version__)"
```

Run this in Colab to confirm GPU:

```python
import torch
print(torch.cuda.is_available())       # True
print(torch.cuda.get_device_name(0))   # Tesla T4 (or similar)
```

---

## Troubleshooting

**Docker fails to start on Windows 11**  
→ Open PowerShell as Administrator and run `wsl --install`, then restart your PC.

**`claude` command not found**  
→ Restart your terminal after `npm install -g`. Make sure Node.js installed correctly.

**Colab disconnects during training**  
→ Enable "Stay connected" in Colab settings, or save checkpoints every 5 epochs to Drive so you can resume.
