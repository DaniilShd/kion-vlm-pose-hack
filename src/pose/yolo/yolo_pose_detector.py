import cv2
import json
import os
import yaml
import time
import logging
from datetime import datetime
from ultralytics import YOLO
from pathlib import Path
import shutil
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.pose.utils.visualization import draw_skeleton

class PoseDetector:
    def __init__(self):
        # загружаем конфиги
        with open("config/pose_config.yaml", 'r') as f:
            self.cfg = yaml.safe_load(f)
        with open("config/paths_config.yaml", 'r') as f:
            self.paths = yaml.safe_load(f)
        
        # создаем папки
        for path in self.paths.values():
            os.makedirs(path, exist_ok=True)
        
        # настройка логирования
        log_dir = self.paths['logs']
        logging.basicConfig(
            filename=f'{log_dir}/pose_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # загружаем модель
        model_name = self.cfg['model']['name']
        self.model_path = os.path.join(self.paths['models'], model_name)
        self.frame_step = self.cfg['video']['frame_step']
        
        if os.path.exists(self.model_path):
            self.model = YOLO(self.model_path)
        else:
            self.model = YOLO(model_name)
            if os.path.exists(model_name):
                shutil.move(model_name, self.model_path)
    
    def process_video(self, video_path):
        video_name = Path(video_path).stem
        out_dir = self.paths['results']
        
        # открываем видео
        cap = cv2.VideoCapture(video_path)
        fps_video = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps_video
        
        print(f"\nВидео: {total_frames} кадров, {fps_video:.2f} fps, {duration:.2f} сек")
        print(f"Обрабатываем каждый {self.frame_step}-й кадр")
        
        # для записи
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_video = os.path.join(out_dir, f"{video_name}_pose.mp4")
        writer = cv2.VideoWriter(out_video, fourcc, fps_video, (w, h))
        
        # данные
        all_frames = []
        frame_num = 0
        processed = 0
        start_time = time.time()
        frame_times = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_num % self.frame_step == 0:
                frame_start = time.time()
                
                results = self.model(frame, conf=self.cfg['model']['confidence'], verbose=False)
                frame_times.append(time.time() - frame_start)
                
                if results[0].keypoints is not None:
                    kps = results[0].keypoints.xyn.cpu().numpy()
                    
                    frame_data = {'frame': frame_num, 'people': []}
                    for i in range(len(kps)):
                        frame_data['people'].append({'keypoints': kps[i].tolist()})
                    all_frames.append(frame_data)
                    
                    frame = draw_skeleton(frame, kps)
                
                processed += 1
                
                if processed % self.cfg['logging']['print_interval'] == 0:
                    print(f"Обработано {processed} кадров")
            
            writer.write(frame)
            frame_num += 1
        
        # статистика
        total_time = time.time() - start_time
        processing_fps = processed / total_time
        effective_fps = processing_fps * self.frame_step
        
        log_data = {
            'video': video_name,
            'total_frames': total_frames,
            'processed_frames': processed,
            'video_fps': round(fps_video, 2),
            'processing_fps': round(processing_fps, 2),
            'effective_fps': round(effective_fps, 2),
            'video_duration': round(duration, 2),
            'processing_time': round(total_time, 2),
            'speedup': round(duration / total_time, 2) if total_time > 0 else 0,
            'frame_step': self.frame_step
        }
        
        self.logger.info(f"Video: {video_name}, Stats: {log_data}")
        
        # вывод
        print(f"\nРЕЗУЛЬТАТЫ")
        print(f"Видео: {duration:.2f} сек, {fps_video:.2f} fps")
        print(f"Обработка: {total_time:.2f} сек")
        print(f"  • FPS модели: {processing_fps:.2f}")
        print(f"  • Эффективный FPS: {effective_fps:.2f}")
        print(f"Ускорение: {log_data['speedup']:.2f}x")
        print(f"Кадров обработано: {processed}/{total_frames}")
        
        # сохраняем ключевые точки в бинарном формате
        npy_path = os.path.join(out_dir, f"{video_name}_keypoints.npy")
        np.save(npy_path, all_frames)  # all_frames как numpy array

        # в JSON только метаданные
        json_path = os.path.join(out_dir, f"{video_name}_meta.json")
        with open(json_path, 'w') as f:
            json.dump({
                'video': video_name,
                'fps': fps_video,
                'frame_step': self.frame_step,
                'processed_frames': processed,
                'keypoints_file': f"{video_name}_keypoints.npy"
            }, f)
        
        cap.release()
        writer.release()
        print(f"\nJSON сохранен: {json_path}")
        print(f"Видео сохранено: {out_video}")
        
        return json_path, out_video, log_data

def main():
    detector = PoseDetector()
    test_dir = detector.paths['test_videos']
    
    if os.path.exists(test_dir):
        videos = [f for f in os.listdir(test_dir) if f.endswith(('.mp4', '.avi'))]
        for v in videos:
            print(f"\n--- {v} ---")
            detector.process_video(os.path.join(test_dir, v))
    else:
        print(f"Нет папки {test_dir}")

if __name__ == "__main__":
    main()