import numpy as np
import json
import os
import glob
import logging
import yaml
from datetime import datetime

class PoseClassifier:
    def __init__(self, paths_config="config/paths_config.yaml"):
        # загружаем пути
        with open(paths_config, 'r') as f:
            self.paths = yaml.safe_load(f)
        
        # создаем папку для логов
        log_dir = self.paths.get('logs', 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        # настройка логирования
        logging.basicConfig(
            filename=f'{log_dir}/classifier_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # индексы ключевых точек YOLO Pose
        self.IDX = {
            'nose': 0,
            'left_eye': 1, 'right_eye': 2,
            'left_ear': 3, 'right_ear': 4,
            'left_shoulder': 5, 'right_shoulder': 6,
            'left_elbow': 7, 'right_elbow': 8,
            'left_wrist': 9, 'right_wrist': 10,
            'left_hip': 11, 'right_hip': 12,
            'left_knee': 13, 'right_knee': 14,
            'left_ankle': 15, 'right_ankle': 16
        }
        
    def distance(self, p1, p2):
        """Евклидово расстояние"""
        return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5
    
    def get_shoulder_width(self, kps):
        """Ширина плеч для нормализации"""
        left = kps[self.IDX['left_shoulder']]
        right = kps[self.IDX['right_shoulder']]
        return self.distance(left, right)
    
    def check_smoking(self, kps):
        """Курение: рука близко к носу (относительно плеч)"""
        nose = kps[self.IDX['nose']]
        left_wrist = kps[self.IDX['left_wrist']]
        right_wrist = kps[self.IDX['right_wrist']]
        shoulder_width = self.get_shoulder_width(kps)
        
        if shoulder_width == 0:
            return False
        
        dist_left = self.distance(nose, left_wrist) / shoulder_width
        dist_right = self.distance(nose, right_wrist) / shoulder_width
        
        return dist_left < 0.5 or dist_right < 0.5
    
    def check_fighting(self, kps1, kps2):
        """Драка: люди близко друг к другу"""
        center1 = np.mean(kps1, axis=0)
        center2 = np.mean(kps2, axis=0)
        
        dist = self.distance(center1, center2)
        
        shoulder1 = self.get_shoulder_width(kps1)
        shoulder2 = self.get_shoulder_width(kps2)
        avg_shoulder = (shoulder1 + shoulder2) / 2
        
        return dist < avg_shoulder * 3
    
    def check_sexual(self, kps1, kps2):
        """Сексуальный контакт: очень близко + характерная поза"""
        center1 = np.mean(kps1, axis=0)
        center2 = np.mean(kps2, axis=0)
        
        dist = self.distance(center1, center2)
        
        shoulder1 = self.get_shoulder_width(kps1)
        shoulder2 = self.get_shoulder_width(kps2)
        avg_shoulder = (shoulder1 + shoulder2) / 2
        
        # очень близко
        if dist > avg_shoulder * 1.5:
            return False
        
        # проверка позы "сзади" (один выше другого по Y)
        hip1 = (kps1[self.IDX['left_hip']] + kps1[self.IDX['right_hip']]) / 2
        hip2 = (kps2[self.IDX['left_hip']] + kps2[self.IDX['right_hip']]) / 2
        
        return abs(hip1[1] - hip2[1]) > avg_shoulder * 0.5
    
    def analyze_video(self, npy_path, meta_path):
        """Анализирует видео"""
        self.logger.info(f"Анализ видео: {npy_path}")
        
        frames = np.load(npy_path, allow_pickle=True)
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        
        suspicious = []
        
        for i, frame in enumerate(frames):
            try:
                # получаем людей на кадре
                if isinstance(frame, dict):
                    people = frame.get('people', [])
                    frame_num = frame.get('frame', i * meta['frame_step'])
                else:
                    people = frame if isinstance(frame, list) else []
                    frame_num = i * meta['frame_step']
                
                if not people:
                    continue
                
                # преобразуем в массив ключевых точек
                people_kps = []
                for p in people:
                    if isinstance(p, dict) and 'keypoints' in p:
                        kps = np.array(p['keypoints'])
                    else:
                        kps = np.array(p)
                    if len(kps) > 0:
                        people_kps.append(kps)
                
                reasons = []
                
                # проверка курения
                for kps in people_kps:
                    if self.check_smoking(kps):
                        reasons.append('smoking')
                        break
                
                # проверка взаимодействий
                if len(people_kps) >= 2:
                    for j in range(len(people_kps)):
                        for k in range(j+1, len(people_kps)):
                            if self.check_fighting(people_kps[j], people_kps[k]):
                                reasons.append('fighting')
                            if self.check_sexual(people_kps[j], people_kps[k]):
                                reasons.append('sexual')
                
                if reasons:
                    suspicious.append({
                        'frame': int(frame_num),
                        'reasons': list(set(reasons))
                    })
                    
            except Exception as e:
                self.logger.error(f"Ошибка на кадре {i}: {e}")
                continue
        
        result = {
            'video': meta.get('video', 'unknown'),
            'total_frames': meta.get('processed_frames', len(frames)),
            'suspicious_frames': suspicious,
            'suspicious_count': len(suspicious)
        }
        
        return result
    
    def process_all(self):
        """Обрабатывает все видео"""
        results_dir = self.paths['results']
        npy_files = glob.glob(os.path.join(results_dir, "*_keypoints.npy"))
        
        print(f"\nНайдено видео: {len(npy_files)}")
        
        for npy_path in npy_files:
            video_name = os.path.basename(npy_path).replace("_keypoints.npy", "")
            meta_path = os.path.join(results_dir, f"{video_name}_meta.json")
            
            if not os.path.exists(meta_path):
                print(f"  {video_name}: нет meta файла")
                continue
            
            print(f"\n--- {video_name} ---")
            
            try:
                result = self.analyze_video(npy_path, meta_path)
                
                # статистика
                reasons_stats = {}
                for f in result['suspicious_frames']:
                    for r in f['reasons']:
                        reasons_stats[r] = reasons_stats.get(r, 0) + 1
                
                print(f"Кадров: {result['total_frames']}")
                print(f"Подозрительных: {result['suspicious_count']}")
                for reason, count in reasons_stats.items():
                    print(f"  {reason}: {count}")
                
                # сохраняем
                out_path = os.path.join(results_dir, f"{video_name}_suspicious.json")
                with open(out_path, 'w') as f:
                    json.dump(result, f, indent=2)
                    
            except Exception as e:
                print(f"Ошибка: {e}")

if __name__ == "__main__":
    classifier = PoseClassifier()
    classifier.process_all()