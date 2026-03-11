import cv2
import yaml
import numpy as np

# загружаем конфиг
with open("config/pose_config.yaml", 'r') as f:
    cfg = yaml.safe_load(f)

COLORS = cfg['colors']
SKELETON = cfg['skeleton']

def draw_skeleton(frame, keypoints_list):
    h, w = frame.shape[:2]
    
    for person_idx, person_kps in enumerate(keypoints_list):
        # Пропускаем пустые массивы
        if person_kps is None or len(person_kps) == 0:
            continue
            
        color = tuple(COLORS[person_idx % len(COLORS)])
        
        # Проверяем, нормализованные ли координаты
        if person_kps.max() <= 1.0:
            person_kps_pixels = person_kps.copy()
            person_kps_pixels[:, 0] = person_kps[:, 0] * w
            person_kps_pixels[:, 1] = person_kps[:, 1] * h
        else:
            person_kps_pixels = person_kps.copy()
        
        # Рисуем только точки, которые не в углу и не нулевые
        valid_points = []
        for i, (x, y) in enumerate(person_kps_pixels):
            # Проверяем, что точка не углу и не нулевая
            if x > 5 and y > 5:  # игнорируем точки слишком близко к (0,0)
                x, y = int(x), int(y)
                if 0 <= x < w and 0 <= y < h:
                    cv2.circle(frame, (x, y), 4, color, -1)
                    valid_points.append((i, x, y))  # запоминаем индекс и координаты
        
        # Рисуем линии только между валидными точками
        for idx1, idx2 in SKELETON:
            # Проверяем, что индексы в пределах массива
            if idx1 >= len(person_kps_pixels) or idx2 >= len(person_kps_pixels):
                continue
            
            # Получаем координаты
            x1, y1 = person_kps_pixels[idx1]
            x2, y2 = person_kps_pixels[idx2]
            
            # Рисуем линию только если обе точки не в углу
            if (x1 > 5 and y1 > 5 and x2 > 5 and y2 > 5 and
                0 <= x1 < w and 0 <= y1 < h and 
                0 <= x2 < w and 0 <= y2 < h):
                cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
    
    return frame