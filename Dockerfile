FROM python:3.11-slim

WORKDIR /app

# Install PyTorch CPU-only build first (separate step so Docker layer-caches it).
# The whl/cpu index only hosts torch/torchvision; other packages use PyPI.
RUN pip install --no-cache-dir \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining app dependencies.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code.
COPY src/ ./src/
COPY app/ ./app/
COPY checkpoints/ ./checkpoints/

EXPOSE 7860

CMD ["python", "app/gradio_app.py"]
