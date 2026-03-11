FROM nvidia/cuda:13.0.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="/venv/bin:$PATH"
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# системные зависимости
RUN apt-get update && apt-get install -y \
    python3 python3-venv python3-pip python3-dev \
    git ffmpeg \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    build-essential \
    curl wget \
    && rm -rf /var/lib/apt/lists/*

# виртуальное окружение
RUN python3 -m venv /venv

# обновление pip
RUN pip install -U pip setuptools wheel

# PyTorch (совместим с CUDA 13 драйвером)
RUN pip install --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

# OpenMMLab installer
RUN pip install --no-cache-dir openmim

# mmcv (wheel)
RUN mim install "mmcv>=2.0.0"

# mmaction
RUN pip install --no-cache-dir mmaction2

# ускорители inference
RUN pip install --no-cache-dir \
    xformers \
    flash-attn --no-build-isolation || true

WORKDIR /app

# сначала requirements (для кэша)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# bitsandbytes
RUN pip install --no-cache-dir bitsandbytes>=0.48.0

# код проекта
COPY .. /app

ENV PYTHONPATH=/app

# проверка GPU
RUN python - <<'PY'
import torch
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY

CMD ["python"]