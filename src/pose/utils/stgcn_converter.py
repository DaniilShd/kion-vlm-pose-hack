import numpy as np
import json
import os
import torch
from pathlib import Path

class STGCNConverter:
    """
    Конвертирует данные из YOLO-pose в формат для ST-GCN (MMAction2)
    """
    def __init__(self):
        # индексы ключевых точек COCO (как в YOLO)
        # 17 точек: нос, глаза, уши, плечи, локти, запястья, бедра, колени, лодыжки
        self.num_joints = 17
        self.num_channels = 3  # x, y, confidence
        
    def convert_npy_to_stgcn(self, npy_path, meta_path, max_frames=300, max_persons=2):
        """
        Конвертирует .npy файл в формат для ST-GCN
        
        Args:
            npy_path: путь к fighting_keypoints.npy
            meta_path: путь к fighting_meta.json
            max_frames: максимальное количество кадров (обрезаем/дополняем)
            max_persons: максимум людей на кадре
        
        Returns:
            keypoints: numpy array формата (1, T, V, C) или (T, V, C) для одного человека
        """
        # загружаем данные
        frames_data = np.load(npy_path, allow_pickle=True)
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        
        print(f"Загружено кадров: {len(frames_data)}")
        print(f"Step: {meta.get('frame_step', 1)}")
        
        # определяем реальное количество кадров
        num_frames = min(len(frames_data), max_frames)
        
        # создаем массив (кадры, ключевые точки, координаты)
        # формат для ST-GCN: (N, T, V, C) где N=1 (batch)
        keypoints = np.zeros((1, max_frames, self.num_joints, self.num_channels))
        
        # заполняем данными
        for t in range(num_frames):
            frame_data = frames_data[t]
            
            # получаем людей на кадре
            if isinstance(frame_data, dict):
                people = frame_data.get('people', [])
            else:
                people = frame_data if isinstance(frame_data, list) else []
            
            # берем первых max_persons людей
            for p_idx, person in enumerate(people[:max_persons]):
                if isinstance(person, dict) and 'keypoints' in person:
                    kps = np.array(person['keypoints'])  # (17, 2)
                    
                    # заполняем координаты
                    keypoints[0, t, :, :2] = kps
                    
                    # confidence = 1 (у YOLO есть но мы пока упростим)
                    keypoints[0, t, :, 2] = 1.0
                elif isinstance(person, (list, np.ndarray)) and len(person) > 0:
                    kps = np.array(person)
                    if kps.shape == (self.num_joints, 2):
                        keypoints[0, t, :, :2] = kps
                        keypoints[0, t, :, 2] = 1.0
        
        # нормализация координат (опционально)
        # keypoints[..., :2] = keypoints[..., :2] / 256.0  # если нужно
        
        print(f"Итоговый массив: {keypoints.shape}")
        print(f"  N (batch): {keypoints.shape[0]}")
        print(f"  T (frames): {keypoints.shape[1]}")
        print(f"  V (joints): {keypoints.shape[2]}")
        print(f"  C (coords): {keypoints.shape[3]}")
        
        return keypoints
    
    def prepare_for_inference(self, keypoints):
        """
        Подготавливает данные для передачи в модель
        """
        # преобразуем в torch tensor
        if isinstance(keypoints, np.ndarray):
            keypoints = torch.from_numpy(keypoints).float()
        
        # добавляем измерение для графа если нужно
        # ST-GCN ожидает (N, C, T, V) или (N, T, V, C)
        # в зависимости от конфига
        
        return keypoints


# пример использования
if __name__ == "__main__":
    converter = STGCNConverter()
    
    # конвертируем твой файл
    keypoints = converter.convert_npy_to_stgcn(
        "data/results/fighting_keypoints.npy",
        "data/results/fighting_meta.json",
        max_frames=300,  # возьмем первые 300 кадров
        max_persons=2
    )
    
    # сохраняем в формате для ST-GCN
    np.save("data/results/fighting_stgcn.npy", keypoints)
    print("Готово!")