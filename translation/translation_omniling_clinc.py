'''
This code translates CLINC150 from english into russian using OmniLing-V1-8b.
'''
import subprocess
import sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "transformers", "accelerate", "bitsandbytes", "sentencepiece", "tqdm"])
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm
import os
import time

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

def translate_text(text):
    prompt = f"""Translate the following English text to Russian. Output ONLY the translation, do not print anything else.

English: {text}
Russian: """

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=32,               
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

    return translation

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

translation_dict = {}
cache_file = "translation_cache_omni_fixed.json"
if os.path.exists(cache_file):
    with open(cache_file, "r", encoding="utf-8") as f:
        translation_dict = json.load(f)

texts_to_translate = [t for t in unique_texts if t not in translation_dict]

if texts_to_translate:

    for i, text in enumerate(tqdm(texts_to_translate)):
        try:
            translated = translate_text(text)
            translation_dict[text] = translated
        except Exception as e:
            print(f"\nОшибка: {text[:50]}... — {e}")
            translation_dict[text] = text + " [ОШИБКА]"

        if (i + 1) % 10 == 0:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(translation_dict, f, ensure_ascii=False)
            time.sleep(1)

    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(translation_dict, f, ensure_ascii=False)

new_data = {}
for split in splits:
    if split not in data:
        continue
    new_data[split] = []
    for item in data[split]:
        en_text = item[0]
        ru_text = translation_dict.get(en_text, en_text + " [НЕТ ПЕРЕВОДА]")
        if len(item) > 1:
            new_item = [ru_text, item[1]]
        else:
            new_item = [ru_text]
        new_data[split].append(new_item)

output_file = "clinc_oos_RU_omni_fixed.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(new_data, f, ensure_ascii=False)