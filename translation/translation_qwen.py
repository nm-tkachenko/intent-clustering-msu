from tqdm import tqdm
import json

def make_prompt(text):

    prompt = f'''Ты переводчик текста. Твоя задача -- перевести сообщение пользователя с английского языка на русский.
    Сообщение будет приведено после слова СООБЩЕНИЕ.
    При переводе постарайся сохранить общий смысл и стилистику исходного сообщения. В твоём ответе не должно быть никаких рассуждений и комментариев, только перевод.
    СООБЩЕНИЕ: {text}
    ПЕРЕВОД: '''

    return prompt

import bitsandbytes
import torch
import json
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

def apply_qwen(prompt, model, tokenizer):
  messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt}
  ]
  text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False  # Setting enable_thinking=False disables thinking mode
)
  model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

  generated_ids = model.generate(
      **model_inputs,
      max_new_tokens=128
  )
  generated_ids = [
      output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
  ]

  response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
  return response

model_name = "RefalMachine/RuadaptQwen3-8B-Hybrid"

# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_use_double_quant=True,
# )

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    # load_in_8bit=True,
    device_map="auto",
    attn_implementation="sdpa",
)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_name)

with open(f'clinc_qwen.json', 'r', encoding="utf-8") as f:
        for fi_le in f:
            data = json.loads(fi_le)
with open(f'all_scores.json', 'r', encoding="utf-8") as f:
        for fi_le in f:
            scores = json.loads(fi_le)

sts = {elem['source']: elem['score'] for elem in scores}

# splits = ['train', 'val', 'test', 'oos_train', 'oos_val', 'oos_test']
for split in ('oos_train', 'oos_val', 'test', 'oos_test', 'train', 'val'):
  count = 0
  for elem in tqdm(data[split]):
    if 'qwen_translation' not in data[split][count].keys() and sts[elem['text']]<0.2:
        data[split][count]['qwen_translation'] = apply_qwen(make_prompt(elem['text']), model, tokenizer)
    count += 1
    if count%50==0:
        with open('clinc_qwen.json', "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        # break
with open('clinc_qwen.json', "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False)

with open(f'banking_qwen.json', 'r', encoding="utf-8") as f:
        for fi_le in f:
            data = json.loads(fi_le)
with open(f'all_scores_b.json', 'r', encoding="utf-8") as f:
        for fi_le in f:
            scores = json.loads(fi_le)

sts = {elem['source']: elem['score'] for elem in scores}

# splits = ['train', 'test', 'oos_train', 'oos_test']
for split in ('train', 'oos_train', 'test', 'oos_test'):
  count = 0
  for elem in tqdm(data[split]):
    if 'qwen_translation' not in data[split][count].keys() and sts[elem['text']]<0.2:
        data[split][count]['qwen_translation'] = apply_qwen(make_prompt(elem['text']), model, tokenizer)
    count += 1
    if count%50==0:
        with open('banking_qwen.json', "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        # break
with open('banking_qwen.json', "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False)
        