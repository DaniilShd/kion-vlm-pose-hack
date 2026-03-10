import torch
import json
import os
import yaml
from PIL import Image
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
from pathlib import Path
import time

class GLMVisionModel:
    def __init__(self):
        # читаем конфиги
        with open("config/vlm_config.yaml", 'r') as f:
            self.vlm_cfg = yaml.safe_load(f)['vlm']
        with open("config/paths_config.yaml", 'r') as f:
            self.paths = yaml.safe_load(f)
        
        # создаем папки
        os.makedirs(self.vlm_cfg['paths']['cache'], exist_ok=True)
        os.makedirs(self.paths['models'], exist_ok=True)
        
        print("Загружаю GLM-4V-9B...")
        
        # ставим tiktoken (нужен для GLM)
        try:
            import tiktoken
        except ImportError:
            print("Устанавливаю tiktoken...")
            os.system("pip install tiktoken")
        
        # грузим модель
        model_name = self.vlm_cfg['model']['name']
        cache_dir = os.path.join(self.paths['models'], "glm-4v")
        
        print(f"Модель: {model_name}")
        print("Загрузка токенизатора...")
        
        # GLM использует AutoTokenizer, а не AutoProcessor
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True
        )
        
        print("Загрузка модели...")
        
        # загружаем модель
        self.model = AutoModel.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            load_in_4bit=self.vlm_cfg['model']['use_4bit'],
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        
        self.model.eval()
        print("Модель загружена!")
    
    def analyze_image(self, image_input):
        """
        Анализ одного изображения
        """
        start = time.time()
        
        # конвертируем вход
        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
        else:
            import cv2
            if isinstance(image_input, torch.Tensor):
                image_input = image_input.cpu().numpy()
            image = Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB))
        
        # готовим промпт
        prompt = self.vlm_cfg['prompts']['moderation']
        
        # для GLM-4V специфичный формат
        inputs = self.model.build_chat_input(
            query=prompt,
            history=[],
            images=[image]
        ).to(self.model.device)
        
        # генерация
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.vlm_cfg['generation']['max_new_tokens'],
                temperature=self.vlm_cfg['generation']['temperature'],
                top_p=self.vlm_cfg['generation']['top_p'],
                do_sample=self.vlm_cfg['generation']['do_sample'],
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # декодируем ответ
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        # парсим JSON
        try:
            # ищем JSON в ответе
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                result = json.loads(json_str)
            else:
                result = json.loads(response)
            
            result['processing_time'] = round(time.time() - start, 2)
            
            # сохраняем если надо
            if self.vlm_cfg['logging']['save_responses']:
                self._save_result(image_input, response, result)
            
            return result
            
        except Exception as e:
            print(f"Ошибка парсинга: {e}")
            print(f"Ответ: {response}")
            return {
                'description': 'ошибка анализа',
                'has_smoking': False,
                'has_same_gender_intimacy': False,
                'has_violence': False,
                'participants_count': 0,
                'confidence': 'low',
                'error': str(e),
                'raw': response
            }
    
    def _save_result(self, image_input, raw_response, parsed):
        """Сохраняет результат в кэш"""
        if isinstance(image_input, str):
            name = Path(image_input).stem
        else:
            name = f"frame_{int(time.time())}"
        
        cache_file = os.path.join(self.vlm_cfg['paths']['cache'], f"{name}.json")
        
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump({
                'raw': raw_response,
                'parsed': parsed,
                'time': time.time()
            }, f, indent=2, ensure_ascii=False)
    
    def analyze_video_frames(self, video_path, frame_indices=None, sample_rate=15):
        """
        Анализ ключевых кадров из видео
        """
        import cv2
        
        print(f"Анализирую видео: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # какие кадры брать
        if frame_indices is None:
            frame_indices = list(range(0, total_frames, sample_rate))
        
        results = []
        
        for i, frame_num in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            print(f"Кадр {frame_num}/{total_frames} ({i+1}/{len(frame_indices)})")
            
            # анализируем
            result = self.analyze_image(frame)
            result['frame'] = frame_num
            results.append(result)
        
        cap.release()
        
        # сохраняем общий результат
        video_name = Path(video_path).stem
        result_file = os.path.join(self.vlm_cfg['paths']['results'], f"{video_name}_glm.json")
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump({
                'video': video_name,
                'model': 'GLM-4V-9B',
                'frames': results
            }, f, indent=2, ensure_ascii=False)
        
        print(f"Результаты сохранены в {result_file}")
        return results


if __name__ == "__main__":
    # быстрый тест
    model = GLMVisionModel()
    
    # берем первое видео из test_videos
    test_dir = "data/test_videos"
    if os.path.exists(test_dir):
        videos = [f for f in os.listdir(test_dir) if f.endswith(('.mp4', '.avi'))]
        if videos:
            video_path = os.path.join(test_dir, videos[0])
            print(f"\nТестирую на {videos[0]}")
            
            # анализируем первые 3 кадра
            results = model.analyze_video_frames(video_path, frame_indices=[0, 30, 60])
            
            for r in results:
                print(f"\nКадр {r['frame']}:")
                print(f"  {r.get('description', 'нет описания')}")
                print(f"  курение: {r.get('has_smoking', False)}")
                print(f"  ЛГБТ: {r.get('has_same_gender_intimacy', False)}")
                print(f"  насилие: {r.get('has_violence', False)}")