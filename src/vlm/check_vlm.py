import torch
from transformers import AutoTokenizer, AutoModel
from PIL import Image
import cv2
import yaml
import glob
import os

# Load config
with open("config/vlm_config.yaml") as f:
    config = yaml.safe_load(f)['vlm']

model_name = config['model']['name']
model_path = config['model']['local_path']

# Check if model exists
if not os.path.exists(os.path.join(model_path, 'config.json')):
    print(f"Downloading {model_name} to {model_path}...")
    os.makedirs(model_path, exist_ok=True)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name,
        load_in_4bit=config['model']['use_4bit'],
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    
    tokenizer.save_pretrained(model_path)
    model.save_pretrained(model_path)
else:
    print(f"Loading model from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    model = AutoModel.from_pretrained(
        model_path,
        load_in_4bit=config['model']['use_4bit'],
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=True
    )

model.eval()

# Get first video
videos = glob.glob("data/test_videos/*.mp4")
if not videos:
    print("No videos found")
    exit()

# Process first frame
cap = cv2.VideoCapture(videos[0])
ret, frame = cap.read()
cap.release()

if ret:
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    prompt = config['prompts']['moderation']
    
    # Правильный способ для GLM-4V
    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "image": image, "content": prompt}],
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True
    ).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=config['generation']['max_new_tokens'],
            temperature=config['generation']['temperature'],
            top_p=config['generation']['top_p']
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\n" + "="*50)
    print(response)
    print("="*50)