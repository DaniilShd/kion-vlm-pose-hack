#!/usr/bin/env python
"""Проверка готовности Docker контейнера к работе с ML моделями."""

import sys
import logging
from pathlib import Path
from typing import Dict, Any, Tuple
import traceback
from datetime import datetime

import torch
import numpy as np
import cv2
import transformers
import gradio
import psutil

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

class ContainerHealthCheck:
    """Проверка здоровья контейнера с ML окружением."""
    
    def __init__(self):
        self.results: Dict[str, bool] = {}
        self.start_time = datetime.now()
        
    def run_all_checks(self) -> bool:
        """Запускает все проверки и возвращает общий результат."""
        self._print_header("НАЧАЛО ПРОВЕРКИ КОНТЕЙНЕРА")
        
        # Информация о системе
        self._log_system_info()
        
        # Список проверок
        checks = [
            ("CUDA/GPU", self._check_cuda),
            ("OpenMP", self._check_openmp),
            ("Память (RAM/GPU)", self._check_memory),
            ("Место на диске", self._check_disk_space),
            ("Загрузка модели", self._check_model_loading),
            ("Зависимости LLaVA", self._check_llava_deps)
        ]
        
        # Запускаем проверки
        for name, check_func in checks:
            self._run_check(name, check_func)
        
        # Выводим итоги
        return self._print_summary()
    
    def _run_check(self, name: str, check_func) -> None:
        """Запускает отдельную проверку с обработкой ошибок."""
        try:
            logger.info(f"🔍 Проверяю: {name}...")
            result = check_func()
            self.results[name] = result
            status = "✅ Успешно" if result else "❌ Провал"
            logger.info(f"{status} - {name}")
        except Exception as e:
            logger.error(f"💥 Ошибка при проверке {name}: {e}")
            logger.debug(traceback.format_exc())
            self.results[name] = False
    
    def _log_system_info(self):
        """Логирует основную информацию о системе."""
        logger.info(f"🐍 Python: {sys.version.split()[0]}")
        logger.info(f"🔥 PyTorch: {torch.__version__}")
        logger.info(f"🎯 CUDA (сборка): {torch.version.cuda}")
        
    def _check_cuda(self) -> bool:
        """Проверяет доступность GPU и CUDA."""
        logger.info("  CUDA доступна: %s", torch.cuda.is_available())
        
        if not torch.cuda.is_available():
            logger.warning("  ⚠️ Контейнер будет работать на CPU (медленно)")
            return False
        
        # Собираем информацию о GPU
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        cuda_version = torch.version.cuda
        
        logger.info(f"  🖥️  GPU: {gpu_name}")
        logger.info(f"  📊 Память GPU: {gpu_memory:.1f} GB")
        logger.info(f"  🎯 CUDA версия: {cuda_version}")
        
        # Тесты CUDA
        tests = [
            ("создание тензора", self._test_tensor_creation),
            ("умножение матриц", self._test_matrix_multiplication),
            ("обратное распространение", self._test_backprop)
        ]
        
        for name, test_func in tests:
            if not test_func():
                return False
        
        logger.info("  ✅ Все CUDA тесты пройдены")
        return True
    
    def _test_tensor_creation(self) -> bool:
        """Тест создания тензора на GPU."""
        try:
            x = torch.tensor([1.0, 2.0, 3.0]).cuda()
            logger.debug(f"    Тензор создан: {x.cpu().numpy()}")
            return True
        except Exception as e:
            logger.error(f"    ❌ Ошибка создания тензора: {e}")
            return False
    
    def _test_matrix_multiplication(self) -> bool:
        """Тест умножения матриц на GPU."""
        try:
            a = torch.randn(100, 100).cuda()
            b = torch.randn(100, 100).cuda()
            c = a @ b
            logger.debug(f"    Умножение матриц работает: {c.shape}")
            return True
        except Exception as e:
            logger.error(f"    ❌ Ошибка умножения матриц: {e}")
            return False
    
    def _test_backprop(self) -> bool:
        """Тест обратного распространения на GPU."""
        try:
            a = torch.randn(10, 10, requires_grad=True).cuda()
            b = torch.randn(10, 10, requires_grad=True).cuda()
            c = (a @ b).sum()
            c.backward()
            logger.debug(f"    Обратное распространение работает")
            return True
        except Exception as e:
            logger.error(f"    ❌ Ошибка обратного распространения: {e}")
            return False
    
    def _check_openmp(self) -> bool:
        """Проверяет работу OpenMP (важно для OpenCV)."""
        try:
            cv2.setNumThreads(4)
            threads = cv2.getNumThreads()
            logger.info(f"  🧵 Потоков OpenCV: {threads}")
            return True
        except Exception as e:
            logger.error(f"  ❌ Ошибка OpenMP: {e}")
            return False
    
    def _check_memory(self) -> bool:
        """Проверяет доступную память."""
        # Проверка RAM
        mem = psutil.virtual_memory()
        logger.info(f"  💾 RAM: {mem.available/1e9:.1f} ГБ свободно / {mem.total/1e9:.1f} ГБ всего")
        
        # Проверка GPU памяти
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            logger.info(f"  🎮 GPU память: {free/1e9:.1f} ГБ свободно / {total/1e9:.1f} ГБ всего")
            
            # Предупреждение если мало памяти
            if free < 2e9:  # меньше 2GB
                logger.warning("  ⚠️ Мало свободной GPU памяти!")
        
        return True
    
    def _check_disk_space(self) -> bool:
        """Проверяет свободное место на диске."""
        try:
            disk = psutil.disk_usage('/')
            free_gb = disk.free / 1e9
            total_gb = disk.total / 1e9
            
            logger.info(f"  💽 Диск: {free_gb:.1f} ГБ свободно / {total_gb:.1f} ГБ всего")
            
            if free_gb < 10:
                logger.warning("  ⚠️ Мало места! Модели LLaVA 7B требуют ~15 ГБ")
            else:
                logger.info("  ✅ Места достаточно")
            
            return True
        except Exception as e:
            logger.error(f"  ❌ Ошибка проверки диска: {e}")
            return False
    
    def _check_model_loading(self) -> bool:
        """Проверяет загрузку и инференс маленькой модели."""
        try:
            from transformers import AutoModel, AutoTokenizer
            
            logger.info("  Загружаю тестовую модель tiny-random-gpt2...")
            
            # Загружаем маленькую модель для теста
            model_name = "hf-internal-testing/tiny-random-gpt2"
            
            logger.info("    Токенизатор...")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            logger.info("    Модель...")
            model = AutoModel.from_pretrained(model_name)
            
            # Перемещаем на GPU если доступно
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            logger.info(f"    Модель на устройстве: {device}")
            
            # Тестовый инференс
            logger.info("    Запускаю инференс...")
            inputs = tokenizer("Hello, how are you?", return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            logger.info(f"    ✅ Инференс работает! Размер выхода: {outputs.last_hidden_state.shape}")
            return True
            
        except Exception as e:
            logger.error(f"  ❌ Ошибка загрузки модели: {e}")
            logger.debug(traceback.format_exc())
            return False
    
    def _check_llava_deps(self) -> bool:
        """Проверяет зависимости для LLaVA."""
        logger.info("  Проверка зависимостей для LLaVA...")
        all_ok = True
        
        # Проверяем bitsandbytes
        try:
            import bitsandbytes as bnb
            logger.info(f"    ✅ bitsandbytes v{bnb.__version__}")
            
            # Тест bitsandbytes с CUDA
            if torch.cuda.is_available():
                try:
                    linear = bnb.nn.Linear8bitLt(10, 10, has_fp16_weights=False).cuda()
                    x = torch.randn(5, 10).cuda()
                    _ = linear(x)
                    logger.info("    ✅ bitsandbytes работает с CUDA")
                except Exception as e:
                    logger.error(f"    ❌ bitsandbytes не работает с CUDA: {e}")
                    all_ok = False
                    
        except ImportError:
            logger.warning("    ⚠️ bitsandbytes не установлен (нужен для LLaVA)")
            all_ok = False
        
        # Проверяем accelerate
        try:
            from accelerate import Accelerator
            accelerator = Accelerator()
            logger.info(f"    ✅ accelerate доступен")
        except Exception as e:
            logger.error(f"    ❌ accelerate: {e}")
            all_ok = False
        
        return all_ok
    
    def _print_header(self, title: str):
        """Печатает красивый заголовок."""
        line = "=" * 60
        print(f"\n{line}")
        print(f"🚀 {title}")
        print(line)
    
    def _print_summary(self) -> bool:
        """Печатает итоги проверки."""
        self._print_header("ИТОГИ ПРОВЕРКИ")
        
        all_passed = True
        for name, passed in self.results.items():
            status = "✅" if passed else "❌"
            print(f"{status} {name}")
            if not passed:
                all_passed = False
        
        # Время выполнения
        elapsed = datetime.now() - self.start_time
        print(f"\n⏱️  Время проверки: {elapsed.total_seconds():.1f} сек")
        
        print("=" * 60)
        
        if all_passed:
            print("✅ КОНТЕЙНЕР ГОТОВ К РАБОТЕ!")
            return True
        else:
            print("⚠️ ЕСТЬ ПРОБЛЕМЫ:")
            if not self.results.get("CUDA/GPU", False):
                print("  • Проверьте драйверы NVIDIA и CUDA в контейнере")
            if not self.results.get("Загрузка модели", False):
                print("  • Проблема с загрузкой моделей (интернет? права?)")
            if not self.results.get("Зависимости LLaVA", False):
                print("  • Установите зависимости для LLaVA")
            return False


def main():
    """Точка входа."""
    checker = ContainerHealthCheck()
    success = checker.run_all_checks()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())