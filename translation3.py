# -*- coding: utf-8 -*-
"""Выборка и перевод 2700 out-of-scope вопросов из natural_questions_clean"""

import subprocess
import sys

print("📦 Устанавливаем зависимости...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "transformers", "accelerate", "bitsandbytes", "sentencepiece", "tqdm", "datasets"])

import json
import torch
import random
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm
import os
import time
from datasets import load_dataset

# ===== ПАРАМЕТРЫ =====
NUM_SAMPLES = 2700          # ← ИЗМЕНЕНО с 1500 на 2700
RANDOM_SEED = 42            # для воспроизводимости
OUTPUT_FILE = "out_of_scope_questions_ru.json"
CACHE_FILE = "translation_cache_oos.json"
# =====================

# --- ШАГ 1: Загружаем датасет и собираем уникальные вопросы ---
print("\n📥 Загружаем natural_questions_clean...")
dataset = load_dataset("rojagtap/natural_questions_clean", split='train', trust_remote_code=True)

# Извлекаем все вопросы, убираем пустые
all_questions = [item['question'] for item in dataset if item.get('question')]
unique_questions = list(set(all_questions))  # уникальные
print(f"✅ Всего уникальных вопросов в датасете: {len(unique_questions)}")

# Случайная выборка
random.seed(RANDOM_SEED)
selected_questions = random.sample(unique_questions, min(NUM_SAMPLES, len(unique_questions)))
print(f"✅ Отобрано {len(selected_questions)} вопросов для перевода")

# --- ШАГ 2: Загружаем модель OmniLing-V1-8b (8-bit) ---
print("\n🌍 Загружаем OmniLing-V1-8b (8-bit)...")
model_name = "WoonaAI/OmniLing-V1-8b-experimental"

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_compute_dtype=torch.float16,
    bnb_8bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)
print("✅ Модель загружена!")

# --- ШАГ 3: Функция перевода (только перевод) ---
def translate_text(text):
    prompt = f"""Translate the following English text to Russian. Output ONLY the translation, do not print anything else.

English: {text}
Russian: """

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,
            temperature=0.1,
            do_sample=False,
            repetition_penalty=1.2,
            early_stopping=True,
            num_beams=1
        )

    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "Russian:" in full_output:
        translation = full_output.split("Russian:")[-1].strip()
    else:
        translation = full_output.strip()

    # Очистка от артефактов (цифры в начале и т.д.)
    import re
    match = re.search(r'[а-яА-ЯёЁa-zA-Z]', translation)
    if match:
        translation = translation[match.start():]

    return translation.strip()

# --- ШАГ 4: Загружаем кэш переводов (чтобы не переводить заново) ---
translation_dict = {}
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        translation_dict = json.load(f)
    print(f"🔄 Загружено {len(translation_dict)} переводов из кэша")

questions_to_translate = [q for q in selected_questions if q not in translation_dict]
print(f"Осталось перевести: {len(questions_to_translate)}")

# --- ШАГ 5: ПЕРЕВОДИМ ---
if questions_to_translate:
    print("\n⏳ Начинаем перевод...")
    for i, q in enumerate(tqdm(questions_to_translate)):
        try:
            translated = translate_text(q)
            translation_dict[q] = translated
        except Exception as e:
            print(f"\n❌ Ошибка: {q[:50]}... — {e}")
            translation_dict[q] = q + " [⚠️ ОШИБКА]"

        # Сохраняем кэш каждые 10 переводов
        if (i + 1) % 10 == 0:
            with open(CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(translation_dict, f, ensure_ascii=False)
            time.sleep(1)

    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(translation_dict, f, ensure_ascii=False)

print(f"\n✅ Всего переведено: {len(translation_dict)}")

# --- ШАГ 6: Сохраняем только переведённые вопросы (список) ---
translated_questions = [translation_dict[q] for q in selected_questions if q in translation_dict]
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(translated_questions, f, ensure_ascii=False)

print(f"\n🎉 Файл с {len(translated_questions)} out-of-scope вопросами сохранён: {OUTPUT_FILE}")