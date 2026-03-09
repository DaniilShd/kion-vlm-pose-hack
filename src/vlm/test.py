#!/usr/bin/env python
"""Минимальная проверка контейнера для работы с ML моделями."""

import sys
import torch
import numpy as np
import cv2
import transformers
import gradio

def check_cuda():
    """Проверка GPU и CUDA."""
    print("\n🔍 ПРОВЕРКА CUDA:")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
        
        # Простой тест тензора на GPU
        x = torch.randn(3, 3).cuda()
        y = torch.randn(3, 3).cuda()
        z = (x @ y).cpu()
        print(f"✅ Тензорное умножение на GPU работает: {z.shape}")
        return True
    else:
        print("❌ CUDA НЕ ДОСТУПНА! Контейнер будет работать на CPU (очень медленно)")
        return False

def check_openmp():
    """Проверка OpenMP (важно для OpenCV)."""
    print("\n🔍 ПРОВЕРКА OPENMP:")
    try:
        cv2.setNumThreads(4)
        threads = cv2.getNumThreads()
        print(f"✅ OpenCV threads: {threads}")
        return True
    except Exception as e:
        print(f"❌ Ошибка OpenMP: {e}")
        return False

def check_memory():
    """Проверка доступной памяти."""
    print("\n🔍 ПРОВЕРКА ПАМЯТИ:")
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        print(f"GPU Memory: {free/1e9:.2f}GB free / {total/1e9:.2f}GB total")
    
    # Проверка RAM
    import psutil
    mem = psutil.virtual_memory()
    print(f"RAM: {mem.available/1e9:.2f}GB available / {mem.total/1e9:.2f}GB total")
    
    return True

def check_tiny_model():
    """Загрузка микро-модели для проверки."""
    print("\n🔍 ПРОВЕРКА ЗАГРУЗКИ МОДЕЛИ:")
    try:
        from transformers import AutoModel, AutoTokenizer
        
        print("Загружаем tiny модель (чтобы проверить transformers)...")
        model_name = "hf-internal-testing/tiny-random-gpt2"
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        
        # Перемещаем на GPU если есть
        if torch.cuda.is_available():
            model = model.cuda()
            print("✅ Модель загружена и перемещена на GPU")
        else:
            print("✅ Модель загружена (CPU)")
        
        # Простой тест
        inputs = tokenizer("Hello", return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        outputs = model(**inputs)
        print(f"✅ Инференс работает: {outputs.last_hidden_state.shape}")
        
        return True
    except Exception as e:
        print(f"❌ Ошибка загрузки модели: {e}")
        return False

def check_llava_install():
    """Проверка что LLaVA можно будет установить."""
    print("\n🔍 ПРОВЕРКА LLAVA ЗАВИСИМОСТЕЙ:")
    try:
        import bitsandbytes
        print(f"✅ bitsandbytes: {bitsandbytes.__version__}")
    except ImportError:
        print("⚠️ bitsandbytes не установлен (будет нужен для LLaVA)")
    
    try:
        from accelerate import Accelerator
        acc = Accelerator()
        print(f"✅ accelerate: доступен")
    except Exception as e:
        print(f"⚠️ accelerate: {e}")
    
    return True

def check_disk_space():
    """Проверка места на диске (для скачивания моделей)."""
    print("\n🔍 ПРОВЕРКА ДИСКА:")
    import shutil
    
    total, used, free = shutil.disk_usage("/")
    print(f"Disk: {free/1e9:.2f}GB free / {total/1e9:.2f}GB total")
    
    if free < 10e9:  # меньше 10GB
        print("⚠️ МАЛО МЕСТА! Модели LLaVA 7B весят ~15GB")
    else:
        print("✅ Места достаточно")
    
    return True

def main():
    """Главная функция проверки."""
    print("=" * 50)
    print("🚀 ПРОВЕРКА DOCKER КОНТЕЙНЕРА ДЛЯ ML")
    print("=" * 50)
    
    # Проверки
    checks = [
        ("CUDA/GPU", check_cuda),
        ("OpenMP", check_openmp),
        ("Memory", check_memory),
        ("Disk", check_disk_space),
        ("Tiny Model", check_tiny_model),
        ("LLaVA Deps", check_llava_install)
    ]
    
    results = []
    for name, func in checks:
        try:
            print(f"\n📌 {name}...")
            result = func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ Ошибка при проверке {name}: {e}")
            results.append((name, False))
    
    # Итог
    print("\n" + "=" * 50)
    print("📊 ИТОГИ ПРОВЕРКИ:")
    print("=" * 50)
    
    all_ok = True
    for name, ok in results:
        status = "✅ OK" if ok else "❌ FAIL"
        print(f"{status} - {name}")
        if not ok:
            all_ok = False
    
    print("=" * 50)
    if all_ok:
        print("✅ КОНТЕЙНЕР ГОТОВ К РАБОТЕ!")
    else:
        print("⚠️ ЕСТЬ ПРОБЛЕМЫ, но основные компоненты могут работать")
    
    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())