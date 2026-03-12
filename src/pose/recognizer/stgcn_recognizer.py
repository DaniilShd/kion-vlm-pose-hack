import torch
import numpy as np
import mmcv
from mmaction.apis import init_recognizer, inference_recognizer
from mmaction.datasets import build_dataset
import os
import yaml
import json
import time
import logging
from datetime import datetime
from ..utils.stgcn_converter import STGCNConverter

class STGCNRecognizer:
    """
    Распознавание действий через ST-GCN (MMAction2)
    """
    def __init__(self, config_path="config/stgcn_ntu60_2d.py", 
                 checkpoint_path="models/stgcn_ntu60_2d.pth"):
        
        # загружаем конфиги
        with open("config/paths_config.yaml", 'r') as f:
            self.paths = yaml.safe_load(f)
        
        # настройка логирования
        log_dir = self.paths['logs']
        os.makedirs(log_dir, exist_ok=True)
        logging.basicConfig(
            filename=f'{log_dir}/stgcn_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        self.converter = STGCNConverter()
        
        # маппинг классов NTU-60 на твои категории
        self.action_map = {
            # индивидуальные
            'sit down': 'sitting',
            'sitting': 'sitting',
            'stand up': 'standing',
            'walking': 'walking',
            'jump up': 'jumping',
            'smoking': 'smoking',
            
            # групповые
            'kicking': 'fighting',
            'punching': 'fighting',
            'stamping': 'fighting',
            'handshaking': 'handshake',
            'hugging': 'hugging',
            'dancing': 'dancing',
            
            # специфические
            'make a phone call': 'meeting',  # замена для митинга
            'cheer up': 'meeting',           # замена для митинга
        }
        
        # загружаем модель
        print(f"Загрузка ST-GCN из {checkpoint_path}")
        self.logger.info(f"Загрузка модели из {checkpoint_path}")
        
        start_time = time.time()
        self.model = init_recognizer(
            config_path,
            checkpoint_path,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        load_time = time.time() - start_time
        
        self.logger.info(f"Модель загружена за {load_time:.2f} сек")
        print(f"Модель загружена за {load_time:.2f} сек!")
    
    def recognize_from_npy(self, npy_path, meta_path):
        """
        Распознает действия по .npy файлу с ключевыми точками
        """
        self.logger.info(f"Распознавание из {npy_path}")
        
        # размер файла
        file_size = os.path.getsize(npy_path) / (1024 * 1024)  # в MB
        self.logger.info(f"Размер файла: {file_size:.2f} MB")
        
        start_time = time.time()
        
        # конвертируем в формат ST-GCN
        keypoints = self.converter.convert_npy_to_stgcn(
            npy_path, 
            meta_path,
            max_frames=300,
            max_persons=2
        )
        
        convert_time = time.time() - start_time
        self.logger.info(f"Конвертация за {convert_time:.2f} сек")
        
        # подготавливаем данные для модели
        data = [keypoints]
        
        # инференс
        inference_start = time.time()
        results = inference_recognizer(self.model, data)
        inference_time = time.time() - inference_start
        
        self.logger.info(f"Инференс за {inference_time:.2f} сек")
        
        # обрабатываем результаты
        actions = self._process_results(results)
        
        total_time = time.time() - start_time
        self.logger.info(f"Всего обработано за {total_time:.2f} сек, найдено {len(actions)} действий")
        
        return actions
    
    def _process_results(self, results):
        """
        Обрабатывает результаты от модели
        """
        if not results:
            self.logger.warning("Нет результатов от модели")
            return []
        
        # берем топ-5 предсказаний
        top5 = results[0][:5]
        
        actions = []
        for class_name, score in top5:
            mapped = self.action_map.get(class_name.lower(), 'other')
            actions.append({
                'original_class': class_name,
                'mapped_action': mapped,
                'confidence': float(score)
            })
            self.logger.debug(f"Действие: {class_name} -> {mapped} ({score:.3f})")
        
        return actions
    
    def analyze_video(self, video_name):
        """
        Анализирует видео по имени (ищет соответствующие .npy и .json)
        """
        base_path = f"data/results/{video_name}"
        npy_path = f"{base_path}_keypoints.npy"
        meta_path = f"{base_path}_meta.json"
        
        self.logger.info(f"Начало анализа видео: {video_name}")
        
        if not os.path.exists(npy_path) or not os.path.exists(meta_path):
            error_msg = f"Файлы не найдены для {video_name}"
            print(error_msg)
            self.logger.error(error_msg)
            return None
        
        print(f"\n--- Анализ {video_name} ---")
        start_time = time.time()
        
        actions = self.recognize_from_npy(npy_path, meta_path)
        
        # группируем по нашим классам
        summary = {}
        for a in actions:
            action = a['mapped_action']
            if action not in summary:
                summary[action] = []
            summary[action].append(a)
        
        print("\nОбнаруженные действия:")
        for action, items in summary.items():
            top_conf = max(items, key=lambda x: x['confidence'])
            print(f"  {action}: {top_conf['confidence']:.2f} ({top_conf['original_class']})")
        
        total_time = time.time() - start_time
        
        # логируем результаты
        self.logger.info(f"Видео: {video_name}")
        self.logger.info(f"Время анализа: {total_time:.2f} сек")
        self.logger.info(f"Найдено действий: {len(actions)}")
        self.logger.info(f"Уникальные классы: {list(summary.keys())}")
        
        # статистика по уверенности
        confidences = [a['confidence'] for a in actions]
        if confidences:
            self.logger.info(f"Средняя уверенность: {np.mean(confidences):.3f}")
            self.logger.info(f"Макс уверенность: {np.max(confidences):.3f}")
        
        print(f"\nВремя анализа: {total_time:.2f} сек")
        
        return {
            'video': video_name,
            'actions': actions,
            'summary': summary,
            'processing_time': total_time
        }


# пример использования
if __name__ == "__main__":
    # инициализация
    recognizer = STGCNRecognizer()
    
    # анализируем видео
    videos = ['fighting', 'smoke', 'trailer_re9']
    all_results = {}
    
    for video in videos:
        result = recognizer.analyze_video(video)
        if result:
            all_results[video] = result
            print(json.dumps(result, indent=2))
    
    # итоговая статистика
    print("\n" + "="*50)
    print("ИТОГОВАЯ СТАТИСТИКА")
    print("="*50)
    
    for video, result in all_results.items():
        print(f"\n{video}:")
        print(f"  Время: {result['processing_time']:.2f} сек")
        print(f"  Действий: {len(result['actions'])}")
        for action, items in result['summary'].items():
            print(f"    {action}: {len(items)}")