import torch
import numpy as np
import mmcv
from mmaction.apis import init_recognizer, inference_recognizer
from mmaction.datasets import build_dataset
import os
import yaml
from .stgcn_converter import STGCNConverter

class STGCNRecognizer:
    """
    Распознавание действий через ST-GCN (MMAction2)
    """
    def __init__(self, config_path="config/stgcn_ntu60_2d.py", 
                 checkpoint_path="models/stgcn_ntu60_2d.pth"):
        
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
        self.model = init_recognizer(
            config_path,
            checkpoint_path,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        print("Модель загружена!")
    
    def recognize_from_npy(self, npy_path, meta_path):
        """
        Распознает действия по .npy файлу с ключевыми точками
        """
        # конвертируем в формат ST-GCN
        keypoints = self.converter.convert_npy_to_stgcn(
            npy_path, 
            meta_path,
            max_frames=300,
            max_persons=2
        )
        
        # подготавливаем данные для модели
        # ST-GCN в MMAction2 ожидает список
        data = [keypoints]
        
        # инференс
        results = inference_recognizer(self.model, data)
        
        # обрабатываем результаты
        return self._process_results(results)
    
    def recognize_from_array(self, keypoints_array):
        """
        Распознает по уже сконвертированному массиву
        """
        data = [keypoints_array]
        results = inference_recognizer(self.model, data)
        return self._process_results(results)
    
    def _process_results(self, results):
        """
        Обрабатывает результаты от модели
        """
        # results - список словарей с вероятностями
        if not results:
            return []
        
        # берем топ-5 предсказаний
        top5 = results[0][:5]
        
        actions = []
        for class_name, score in top5:
            # маппим на наши классы
            mapped = self.action_map.get(class_name.lower(), 'other')
            actions.append({
                'original_class': class_name,
                'mapped_action': mapped,
                'confidence': float(score)
            })
        
        return actions
    
    def analyze_video(self, video_name):
        """
        Анализирует видео по имени (ищет соответствующие .npy и .json)
        """
        base_path = f"data/results/{video_name}"
        npy_path = f"{base_path}_keypoints.npy"
        meta_path = f"{base_path}_meta.json"
        
        if not os.path.exists(npy_path) or not os.path.exists(meta_path):
            print(f"Файлы не найдены для {video_name}")
            return None
        
        print(f"\n--- Анализ {video_name} ---")
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
        
        return {
            'video': video_name,
            'actions': actions,
            'summary': summary
        }


# пример использования
if __name__ == "__main__":
    # инициализация
    recognizer = STGCNRecognizer()
    
    # анализируем видео
    for video in ['fighting', 'smoke', 'trailer_re9']:
        result = recognizer.analyze_video(video)
        if result:
            print(json.dumps(result, indent=2))