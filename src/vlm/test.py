#!/usr/bin/env python
"""Минимальная проверка контейнера для работы с ML моделями."""

import sys
import torch
import numpy as np
import cv2
import transformers
import gradio
import traceback

def check_cuda():
    """Проверка GPU и CUDA с детальной диагностикой."""
    print("\n🔍 ПРОВЕРКА CUDA:")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if not torch.cuda.is_available():
        print("❌ CUDA НЕ ДОСТУПНА! Контейнер будет работать на CPU (очень медленно)")
        return False
    
    try:
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
        
        # Тест 1: Простое создание тензора на GPU
        print("  Тест 1: Создание тензора на GPU...")
        x = torch.tensor([1.0, 2.0, 3.0]).cuda()
        print(f"    ✅ Тензор создан: {x.cpu().numpy()}")
        
        # Тест 2: Простое умножение матриц
        print("  Тест 2: Умножение матриц...")
        a = torch.randn(3, 3).cuda()
        b = torch.randn(3, 3).cuda()
        c = torch.mm(a, b)
        print(f"    ✅ Умножение работает: {c.shape}")
        
        # Тест 3: Операции с градиентами
        print("  Тест 3: Операции с градиентами...")
        a = torch.randn(3, 3, requires_grad=True).cuda()
        b = torch.randn(3, 3, requires_grad=True).cuda()
        c = torch.mm(a, b)
        loss = c.sum()
        loss.backward()
        print(f"    ✅ Обратное распространение работает")
        
        print("✅ Все CUDA тесты пройдены успешно!")
        return True
        
    except Exception as e:
        print(f"❌ Ошибка при работе с CUDA: {e}")
        print("\nДетальная диагностика:")
        traceback.print_exc()
        
        # Дополнительная диагностика
        print("\n📊 Дополнительная информация:")
        print(f"  CUDA версия драйвера: {torch.cuda.get_device_properties(0).major}.{torch.cuda.get_device_properties(0).minor}")
        print(f"  Total memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"  Multiprocessors: {torch.cuda.get_device_properties(0).multi_processor_count}")
        
        # Проверка CUBLAS
        try:
            torch.ones(1).cuda() @ torch.ones(1).cuda()
        except Exception as e:
            print(f"  ❌ CUBLAS ошибка: {e}")
        
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
        traceback.print_exc()
        return False

def check_memory():
    """Проверка доступной памяти."""
    print("\n🔍 ПРОВЕРКА ПАМЯТИ:")
    try:
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            print(f"GPU Memory: {free/1e9:.2f}GB free / {total/1e9:.2f}GB total")
        
        # Проверка RAM
        import psutil
        mem = psutil.virtual_memory()
        print(f"RAM: {mem.available/1e9:.2f}GB available / {mem.total/1e9:.2f}GB total")
        
        return True
    except Exception as e:
        print(f"❌ Ошибка при проверке памяти: {e}")
        traceback.print_exc()
        return False

def check_tiny_model():
    """Загрузка микро-модели для проверки."""
    print("\n🔍 ПРОВЕРКА ЗАГРУЗКИ МОДЕЛИ:")
    try:
        from transformers import AutoModel, AutoTokenizer
        
        print("Загружаем tiny модель (чтобы проверить transformers)...")
        model_name = "hf-internal-testing/tiny-random-gpt2"
        
        print("  Загрузка токенизатора...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        print("  Загрузка модели...")
        model = AutoModel.from_pretrained(model_name)
        
        # Перемещаем на GPU если есть
        if torch.cuda.is_available():
            print("  Перемещение модели на GPU...")
            model = model.cuda()
            print("✅ Модель загружена и перемещена на GPU")
        else:
            print("✅ Модель загружена (CPU)")
        
        # Простой тест
        print("  Подготовка входных данных...")
        inputs = tokenizer("Hello", return_tensors="pt")
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        print("  Запуск инференса...")
        with torch.no_grad():  # Отключаем градиенты для экономии памяти
            outputs = model(**inputs)
        
        print(f"✅ Инференс работает: {outputs.last_hidden_state.shape}")
        return True
        
    except Exception as e:
        print(f"❌ Ошибка загрузки модели: {e}")
        print("\nДетальная диагностика:")
        traceback.print_exc()
        
        # Проверка доступности transformers
        print(f"\nTransformers version: {transformers.__version__}")
        
        return False

def check_llava_install():
    """Проверка что LLaVA можно будет установить."""
    print("\n🔍 ПРОВЕРКА LLAVA ЗАВИСИМОСТЕЙ:")
    all_ok = True
    
    try:
        import bitsandbytes
        print(f"✅ bitsandbytes: {bitsandbytes.__version__}")
        
        # Проверка что bitsandbytes работает с CUDA
        if torch.cuda.is_available():
            try:
                # Простой тест bitsandbytes
                linear = bitsandbytes.nn.Linear8bitLt(10, 10, has_fp16_weights=False).cuda()
                x = torch.randn(5, 10).cuda()
                y = linear(x)
                print("  ✅ bitsandbytes работает с CUDA")
            except Exception as e:
                print(f"  ❌ bitsandbytes ошибка с CUDA: {e}")
                all_ok = False
    except ImportError:
        print("⚠️ bitsandbytes не установлен (будет нужен для LLaVA)")
        all_ok = False
    except Exception as e:
        print(f"❌ Ошибка с bitsandbytes: {e}")
        traceback.print_exc()
        all_ok = False
    
    try:
        from accelerate import Accelerator
        acc = Accelerator()
        print(f"✅ accelerate: доступен")
    except Exception as e:
        print(f"❌ accelerate: {e}")
        traceback.print_exc()
        all_ok = False
    
    return all_ok

def check_disk_space():
    """Проверка места на диске (для скачивания моделей)."""
    print("\n🔍 ПРОВЕРКА ДИСКА:")
    try:
        import shutil
        
        total, used, free = shutil.disk_usage("/")
        print(f"Disk: {free/1e9:.2f}GB free / {total/1e9:.2f}GB total")
        
        if free < 10e9:  # меньше 10GB
            print("⚠️ МАЛО МЕСТА! Модели LLaVA 7B весят ~15GB")
        else:
            print("✅ Места достаточно")
        
        return True
    except Exception as e:
        print(f"❌ Ошибка при проверке диска: {e}")
        traceback.print_exc()
        return False

def main():
    """Главная функция проверки."""
    print("=" * 60)
    print("🚀 ПРОВЕРКА DOCKER КОНТЕЙНЕРА ДЛЯ ML")
    print("=" * 60)
    
    # Информация о системе
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA Build: {torch.version.cuda}")
    
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
            print(f"\n{'='*40}")
            print(f"📌 {name}...")
            print(f"{'='*40}")
            result = func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ Критическая ошибка при проверке {name}: {e}")
            traceback.print_exc()
            results.append((name, False))
    
    # Итог
    print("\n" + "=" * 60)
    print("📊 ИТОГИ ПРОВЕРКИ:")
    print("=" * 60)
    
    all_ok = True
    for name, ok in results:
        status = "✅ OK" if ok else "❌ FAIL"
        print(f"{status} - {name}")
        if not ok:
            all_ok = False
    
    print("=" * 60)
    if all_ok:
        print("✅ КОНТЕЙНЕР ПОЛНОСТЬЮ ГОТОВ К РАБОТЕ!")
    else:
        print("⚠️ ЕСТЬ ПРОБЛЕМЫ, требующие решения")
        print("\nРекомендации:")
        if not results[0][1]:  # CUDA failed
            print("  • Проверьте установку CUDA библиотек в контейнере")
            print("  • Убедитесь что драйвер NVIDIA на хосте поддерживает CUDA 13.0")
        if not results[4][1]:  # Tiny model failed
            print("  • Проблема с загрузкой/инференсом модели")
            print("  • Проверьте transformers и его зависимость от CUDA")
    
    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())