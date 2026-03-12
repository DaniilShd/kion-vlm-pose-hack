#!/usr/bin/env python3
import os
import argparse
from pose.yolo_pose.yolo_pose_detector import PoseDetector

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='видео для обработки')
    args = parser.parse_args()
    
    detector = PoseDetector()
    
    if args.input:
        if os.path.exists(args.input):
            detector.process_video(args.input)
        else:
            print(f"Файл {args.input} не найден")
    else:
        # обрабатываем все видео из test_videos
        test_dir = detector.paths['test_videos']
        if os.path.exists(test_dir):
            videos = [f for f in os.listdir(test_dir) if f.endswith(('.mp4', '.avi'))]
            for v in videos:
                print(f"\n=== {v} ===")
                detector.process_video(os.path.join(test_dir, v))
        else:
            print(f"Нет папки {test_dir}")

if __name__ == "__main__":
    main()