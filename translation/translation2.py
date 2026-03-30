# -*- coding: utf-8 -*-
"""Перевод BANKING77 с OmniLing-V1-8b (Parquet-версия, очистка артефактов, лимит 64 токена)"""

import subprocess
import sys
import re  # для регулярных выражений

print("📦 Устанавливаем зависимости...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "transformers", "accelerate", "bitsandbytes", "sentencepiece", "tqdm", "datasets"])

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm
import os
import time
from datasets import load_dataset

# --- ШАГ 1: Загружаем датасет BANKING77 (Parquet-версия) ---
print("\n📥 Загружаем BANKING77 (Parquet-версия)...")
dataset = load_dataset(
    "PolyAI/banking77",
    revision="refs/convert/parquet",   # специальная ветка с Parquet-файлами
    split=['train', 'test']            # загружаем сразу train и test
)

train_data = dataset[0]
test_data = dataset[1]

# Извлекаем имена интентов из features датасета
intent_names = train_data.features['label'].names
print(f"✅ Количество интентов: {len(intent_names)}")
print(f"Пример интента под номером 0: {intent_names[0]}")

# Создаём структуру данных, аналогичную CLINC150
data = {
    'train': [],
    'val': [],
    'test': [],
    'oos_train': [],
    'oos_val': [],
    'oos_test': []
}

# Заполняем train
for item in train_data:
    text = item['text']
    label_num = item['label']
    intent_name = intent_names[label_num]
    data['train'].append([text, intent_name])

# Заполняем test
for item in test_data:
    text = item['text']
    label_num = item['label']
    intent_name = intent_names[label_num]
    data['test'].append([text, intent_name])

print(f"✅ Загружено {len(data['train'])} тренировочных и {len(data['test'])} тестовых примеров")

# --- ШАГ 2: Загружаем OmniLing-V1-8b с 8-битной квантизацией ---
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

# --- ШАГ 3: Функция перевода с постобработкой (очистка артефактов) ---
def translate_text(text):
    """Переводит текст и удаляет лишние символы в начале."""
    prompt = f"""Translate the following English text to Russian. Output ONLY the translation, do not print anything else.

English: {text}
Russian: """

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,          # ← УВЕЛИЧЕНО С 32 ДО 64
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

    if "\n\n" in translation:
        translation = translation.split("\n\n")[0].strip()

    # Очистка: удаляем все символы до первой буквы (русской или английской)
    match = re.search(r'[а-яА-ЯёЁa-zA-Z]', translation)
    if match:
        translation = translation[match.start():]

    return translation.strip()

# --- ШАГ 4: Собираем уникальные фразы ---
print("\n📖 Собираем уникальные фразы...")
splits = ['train', 'val', 'test', 'oos_train', 'oos_val', 'oos_test']
all_texts = set()
for split in splits:
    if split in data:
        for item in data[split]:
            if isinstance(item, list) and len(item) > 0:
                all_texts.add(item[0])

unique_texts = list(all_texts)
print(f"✅ Найдено уникальных фраз: {len(unique_texts)}")

# --- ШАГ 5: Загружаем кэш ---
translation_dict = {}
cache_file = "translation_cache_banking77.json"
if os.path.exists(cache_file):
    with open(cache_file, "r", encoding="utf-8") as f:
        translation_dict = json.load(f)
    print(f"🔄 Загружено {len(translation_dict)} переводов из кэша")

texts_to_translate = [t for t in unique_texts if t not in translation_dict]
print(f"Осталось перевести: {len(texts_to_translate)}")

# --- ШАГ 6: ПЕРЕВОДИМ ---
if texts_to_translate:
    print("\n⏳ Начинаем перевод...")
    # Для теста можно ограничить:
    # texts_to_translate = texts_to_translate[:20]

    for i, text in enumerate(tqdm(texts_to_translate)):
        try:
            translated = translate_text(text)
            translation_dict[text] = translated
        except Exception as e:
            print(f"\n❌ Ошибка: {text[:50]}... — {e}")
            translation_dict[text] = text + " [⚠️ ОШИБКА]"

        if (i + 1) % 10 == 0:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(translation_dict, f, ensure_ascii=False)
            time.sleep(1)

    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(translation_dict, f, ensure_ascii=False)

print(f"\n✅ Всего переведено: {len(translation_dict)}")

# --- ШАГ 7: Собираем новый датасет ---
print("\n🔄 Формируем русскоязычный датасет...")
new_data = {}
for split in splits:
    if split not in data:
        continue
    new_data[split] = []
    for item in data[split]:
        en_text = item[0]
        ru_text = translation_dict.get(en_text, en_text + " [⚠️ НЕТ ПЕРЕВОДА]")
        if len(item) > 1:
            new_item = [ru_text, item[1]]
        else:
            new_item = [ru_text]
        new_data[split].append(new_item)

# --- ШАГ 8: Сохраняем результат ---
output_file = "banking77_RU_omni_fixed.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(new_data, f, indent=2, ensure_ascii=False)

print(f"\n🎉 Файл сохранён: {output_file}")

# --- ШАГ 9: Скачиваем ---
# from google.colab import files
# files.download(output_file)
# print("\n📥 Если не скачалось, найди файл в панели слева.")