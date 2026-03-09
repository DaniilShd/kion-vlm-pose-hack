FROM nvidia/cuda:13.1.1-runtime-ubuntu22.04

# Устанавливаем Python и системные зависимости
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Создаем ссылку python
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Устанавливаем PyTorch с поддержкой CUDA 12.4
RUN pip3 install --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu130

# Устанавливаем базовые библиотеки
RUN pip3 install --no-cache-dir \
    transformers \
    accelerate \
    gradio \
    fastapi \
    uvicorn \
    opencv-python-headless \
    numpy \
    mmpose \
    huggingface_hub

WORKDIR /app

# Создаем папки для данных
RUN mkdir -p /app/data/{uploads,results} /app/src /app/models

CMD ["bash"]