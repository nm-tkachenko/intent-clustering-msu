# -*- coding: utf-8 -*-
"""Перевод CLINC150 с OmniLing-V1-8b (исправленный промпт, 8-bit)"""

import subprocess
import sys

print("📦 Устанавливаем зависимости...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "transformers", "accelerate", "bitsandbytes", "sentencepiece", "tqdm"])

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm
import os
import time

# --- ШАГ 1: Скачиваем датасет ---
print("\n📥 Загружаем CLINC150...")
# !wget -q -O clinc_oos.json https://raw.githubusercontent.com/clinc/oos-eval/master/data/data_full.json
print("✅ Датасет скачан")

# --- ШАГ 2: Загружаем OmniLing-V1-8b с 8-битной квантизацией (качество лучше, память ~8-9 ГБ) ---
print("\n🌍 Загружаем OmniLing-V1-8b (8-bit)...")
model_name = "WoonaAI/OmniLing-V1-8b-experimental"

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,                     # 8-битная квантизация
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

# --- ШАГ 3: Улучшенная функция перевода ---
def translate_text(text):
    """Переводит текст и строго ограничивает вывод только переводом."""
    # Чёткий промпт: запрещаем лишний текст
    prompt = f"""Translate the following English text to Russian. Output ONLY the translation, do not print anything else.

English: {text}
Russian: """

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=32,               # достаточно для коротких фраз
            temperature=0.1,                   # минимальная случайность
            do_sample=False,                    # отключаем сэмплинг (жадный поиск)
            repetition_penalty=1.2,              # штраф за повторы
            early_stopping=True,
            num_beams=1                          # без beam search (быстрее)
        )

    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Извлекаем только текст после "Russian:"
    if "Russian:" in full_output:
        translation = full_output.split("Russian:")[-1].strip()
    else:
        translation = full_output.strip()

    # Обрезаем, если модель случайно добавила ещё один перевод
    # (иногда начинает генерировать новый пример)
    if "\n\n" in translation:
        translation = translation.split("\n\n")[0].strip()

    return translation

# --- ШАГ 4: Читаем датасет и собираем уникальные фразы ---
print("\n📖 Читаем датасет...")
with open("clinc_oos.json", "r") as f:
    data = json.load(f)

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
cache_file = "translation_cache_omni_fixed.json"
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

        # Сохраняем каждые 10 фраз
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
output_file = "clinc_oos_RU_omni_fixed.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(new_data, f, ensure_ascii=False)

print(f"\n🎉 Файл сохранён: {output_file}")

# --- ШАГ 9: Скачиваем ---
# from google.colab import files
# files.download(output_file)
# print("\n📥 Если не скачалось, найди файл в панели слева.")