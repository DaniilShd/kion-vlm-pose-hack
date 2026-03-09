FROM nvidia/cuda:13.0.0-devel-ubuntu24.04  

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y \
    python3 \
    python3-venv \
    python3-pip \
    python3-dev \
    build-essential \
    git \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Создаем виртуальное окружение
RUN python3 -m venv /venv
ENV PATH="/venv/bin:$PATH"

# Обновляем pip
RUN pip install -U pip setuptools wheel

# Устанавливаем стабильный PyTorch для CUDA 13.0
RUN pip install --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu130 \
    --timeout 1000 \
    --retries 20

# Устанавливаем CUDA библиотеки (КРИТИЧЕСКИ ВАЖНО!)
RUN pip install --no-cache-dir \
    nvidia-cuda-runtime-cu12 \
    nvidia-cublas-cu12 \
    nvidia-cudnn-cu12 \
    nvidia-cuda-nvrtc-cu12

WORKDIR /app

# Копируем requirements
COPY requirements.txt .

# Устанавливаем зависимости
RUN pip install -r requirements.txt

# Копируем остальной код
COPY . /app

# Настраиваем переменные окружения CUDA
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/lib:/usr/local/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}
ENV PATH=/usr/local/cuda/bin:${PATH}
ENV PYTHONPATH=/app

CMD ["tail", "-f", "/dev/null"]