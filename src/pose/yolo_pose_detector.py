import cv2
import json
import os
import yaml
from ultralytics import YOLO
from pathlib import Path
import time
import shutil

class PoseDetector:
    def __init__(self):
        # читаем конфиги
        with open("config/pose_config.yaml", 'r') as f:
            self.cfg = yaml.safe_load(f)
        with open("config/paths_config.yaml", 'r') as f:
            self.paths = yaml.safe_load(f)
        
        # создаем папки
        for path in self.paths.values():
            os.makedirs(path, exist_ok=True)
        
        # путь к модели в папке models
        model_name = self.cfg['model']['name']
        self.model_path = os.path.join(self.paths['models'], model_name)
        
        # загружаем модель
        print(f"Ищу модель в {self.model_path}...")
        
        if os.path.exists(self.model_path):
            print("Модель найдена, загружаю...")
            self.model = YOLO(self.model_path)
        else:
            print("Модель не найдена, скачиваю...")
            # скачиваем
            self.model = YOLO(model_name)
            # сохраняем в нашу папку
            if os.path.exists(model_name):
                shutil.move(model_name, self.model_path)
                print(f"Модель сохранена в {self.model_path}")
        
        print("Модель загружена!")
    
    def process_video(self, video_path):
        video_name = Path(video_path).stem
        out_dir = self.paths['results']
        
        # открываем видео
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Видео: {total} кадров, {fps}fps")
        
        # для записи
        fourcc = cv2.VideoWriter_fourcc(*self.cfg['video']['codec'])
        out_video = os.path.join(out_dir, f"{video_name}_pose.mp4")
        writer = cv2.VideoWriter(out_video, fourcc, fps, (w, h))
        
        # данные для json
        all_frames = []
        frame_num = 0
        start = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # детекция
            results = self.model(frame, conf=self.cfg['model']['confidence'])
            
            # собираем ключевые точки
            frame_data = {'frame': frame_num, 'people': []}
            
            if results[0].keypoints is not None:
                kps = results[0].keypoints.xy.cpu().numpy()
                for i in range(len(kps)):
                    frame_data['people'].append({
                        'keypoints': kps[i].tolist()
                    })
            
            all_frames.append(frame_data)
            
            # сохраняем кадр
            writer.write(results[0].plot())
            
            frame_num += 1
            if frame_num % self.cfg['logging']['print_interval'] == 0:
                print(f"Кадр {frame_num}/{total}")
        
        # сохраняем json
        json_path = os.path.join(out_dir, f"{video_name}.json")
        with open(json_path, 'w') as f:
            json.dump({
                'video': video_name,
                'fps': fps,
                'frames': all_frames
            }, f, indent=2)
        
        cap.release()
        writer.release()
        
        print(f"Готово! Время: {time.time()-start:.1f}с")
        print(f"JSON: {json_path}")
        print(f"Видео: {out_video}")
        
        return json_path, out_video

def main():
    detector = PoseDetector()
    
    # берем видео из test_videos
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