# Минимальный рабочий Dockerfile для RTX 5070 (CUDA 13)
FROM nvidia/cuda:13.0-runtime-ubuntu22.04

# Устанавливаем Python и минимум необходимого
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Создаем ссылку python
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Устанавливаем PyTorch с поддержкой CUDA 13
RUN pip3 install --no-cache-dir \
    --pre torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu130

# Устанавливаем базовые библиотеки для хакатона
RUN pip3 install --no-cache-dir \
    transformers \
    accelerate \
    gradio \
    fastapi \
    uvicorn \
    opencv-python \
    numpy

WORKDIR /app

# Создаем папки для данных
RUN mkdir -p /app/data/{uploads,results}

CMD ["bash"]