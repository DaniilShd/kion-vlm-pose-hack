import cv2
import json
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
from config.config_loader import ConfigLoader

# Загружаем конфиг
loader = ConfigLoader()
pose_cfg = loader.load_pose_config()

COLORS = pose_cfg['visualization']['colors']
SKELETON = pose_cfg['visualization']['skeleton']

def draw_skeleton(frame, keypoints, conf=None, conf_thresh=0.3):
    """
    Рисует скелет на кадре
    """
    h, w = frame.shape[:2]
    
    for i, person_kps in enumerate(keypoints):
        color = COLORS[i % len(COLORS)]
        
        # Рисуем точки
        for j, (x, y) in enumerate(person_kps):
            if conf and conf[i][j] < conf_thresh:
                continue
            x, y = int(x), int(y)
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(frame, (x, y), 3, color, -1)
        
        # Рисуем соединения
        for idx1, idx2 in SKELETON:
            if idx1 >= len(person_kps) or idx2 >= len(person_kps):
                continue
                
            x1, y1 = person_kps[idx1]
            x2, y2 = person_kps[idx2]
            
            if conf and (conf[i][idx1] < conf_thresh or conf[i][idx2] < conf_thresh):
                continue
                
            x1, y1 = int(x1), int(y1)
            x2, y2 = int(x2), int(y2)
            
            if all(0 <= p[0] < w and 0 <= p[1] < h for p in [(x1,y1), (x2,y2)]):
                cv2.line(frame, (x1, y1), (x2, y2), color, 2)

def create_video_from_json(json_path, video_path, output_path=None):
    """
    Создает видео со скелетами из JSON файла
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    if output_path is None:
        output_path = video_path.replace('.mp4', '_skeleton.mp4')
    
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*pose_cfg['video']['codec'])
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    
    frame_num = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_num < len(data['data']):
            frame_data = data['data'][frame_num]
            people = frame_data.get('people', [])
            
            kps_list = []
            conf_list = []
            
            for person in people:
                if 'keypoints' in person:
                    kps_list.append(person['keypoints'])
                    if 'confidence' in person:
                        conf_list.append(person['confidence'])
            
            if kps_list:
                draw_skeleton(frame, kps_list, conf_list if conf_list else None)
        
        out.write(frame)
        frame_num += 1
    
    cap.release()
    out.release()
    print(f"[INFO] Видео сохранено: {output_path}")
    return output_path