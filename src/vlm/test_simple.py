import torch
from transformers import AutoTokenizer, AutoModel
from PIL import Image
import cv2
import json

print("Загружаю модель...")

tokenizer = AutoTokenizer.from_pretrained(
    "THUDM/glm-4v-9b",
    trust_remote_code=True
)

model = AutoModel.from_pretrained(
    "THUDM/glm-4v-9b",
    load_in_4bit=True,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)

model.eval()
print("Модель загружена!")

# берем кадр из видео
cap = cv2.VideoCapture("data/test_videos/smoke.mp4")
ret, frame = cap.read()
cap.release()

if ret:
    # конвертируем в PIL
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    # промпт
    prompt = "Опиши что на картинке. Есть ли курящие люди? Ответь кратко."
    
    # готовим вход
    inputs = model.build_chat_input(
        query=prompt,
        history=[],
        images=[image]
    ).to(model.device)
    
    # генерируем
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.1
        )
    
    # ответ
    response = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:],
        skip_special_tokens=True
    )
    
    print(f"\nОтвет модели: {response}")
else:
    print("Не удалось прочитать видео")