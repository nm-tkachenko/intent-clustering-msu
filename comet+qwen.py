from comet import download_model, load_from_checkpoint

model_path = download_model("Unbabel/wmt20-comet-qe-da")
model = load_from_checkpoint(model_path)

import csv
import json

with open('/content/clinc_qwen.json', 'r', encoding="utf-8") as f:
    data = json.load(f)
with open('all_scores_qwen.csv', 'w', newline='', encoding='utf-8') as f_csv:
    writer = csv.writer(f_csv)
    writer.writerow(['split', 'source', 'translation', 'score'])
    splits = ['train', 'val', 'test', 'oos_train', 'oos_val', 'oos_test']

    for split in splits:
        if split not in data:
            continue

        items = data[split]
        if not items:
            continue

        data_for_model = []
        for item in items:
            src = item.get('text')
            mt = item.get('qwen_translation', 'none')
            if src is not None and mt is not None and src.strip() != "" and mt.strip() != "":
                data_for_model.append({"src": src, "mt": mt})

        if not data_for_model:
            continue

        output = model.predict(data_for_model, batch_size=8, gpus=1)
        scores = output.scores
        mean_score = sum(scores) / len(scores)
        print(f"{split}: средняя оценка = {mean_score:.4f}")

        for item, score in zip(data_for_model, scores):
            writer.writerow([split, item['src'], item['mt'], f"{score:.4f}"])

csv_file = 'all_scores_qwen.csv'
json_file = 'all_scores_qwen.json'

data = []
with open(csv_file, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        row['score'] = float(row['score'])
        data.append(row)

with open(json_file, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False)

import json
from tqdm import tqdm
with open('/content/clinc_qwen.json', 'r', encoding="utf-8") as f:
  data = json.load(f)
with open('/content/all_scores_qwen.json', 'r', encoding="utf-8") as f:
  qscores = json.load(f)
  qsts = {elem['source']: elem['score'] for elem in qscores}
with open('/content/all_scores.json', 'r', encoding="utf-8") as f:
  scores = json.load(f)
  sts = {elem['source']: elem['score'] for elem in scores}
# splits = ['train', 'val', 'test', 'oos_train', 'oos_val', 'oos_test']

# results = {}

for split in splits:
    for item in tqdm(data[split]):
      if sts[item['text']]<qsts[item['text']] and 'qwen_translation' in item:
        item['translation'] = item['qwen_translation']
      elif sts[item['text']]<qsts[item['text']] and 'qwen_translation' not in item:
        print(item)
      if 'qwen_translation' in item:
        del item['qwen_translation']
      item['score'] = max(sts[item['text']], qsts[item['text']])

with open('clinc_qwen2.json', "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False)

for split in data:
  score = sum([elem['score'] for elem in data[split]])/len(data[split])
  print(split, score)

# train 0.42042508666666667
# val 0.4149742666666667
# test 0.4349316
# oos_train 0.530372
# oos_val 0.562052
# oos_test 0.4375181

with open('/content/banking_qwen.json', 'r', encoding="utf-8") as f:
    data = json.load(f)
with open('all_scores_b_qwen.csv', 'w', newline='', encoding='utf-8') as f_csv:
    writer = csv.writer(f_csv)
    writer.writerow(['split', 'source', 'translation', 'score'])
    splits = ['train', 'val', 'test', 'oos_train', 'oos_val', 'oos_test']

    for split in splits:
        if split not in data:
            continue

        items = data[split]
        if not items:
            continue

        data_for_model = []
        for item in items:
            src = item.get('text')
            mt = item.get('qwen_translation', 'none')
            if src is not None and mt is not None and src.strip() != "" and mt.strip() != "":
                data_for_model.append({"src": src, "mt": mt})

        if not data_for_model:
            continue

        output = model.predict(data_for_model, batch_size=8, gpus=1)
        scores = output.scores

        mean_score = sum(scores) / len(scores)
        print(f"{split}: средняя оценка = {mean_score:.4f}")

        for item, score in zip(data_for_model, scores):
            writer.writerow([split, item['src'], item['mt'], f"{score:.4f}"])

csv_file = 'all_scores_b_qwen.csv'
json_file = 'all_scores_b_qwen.json'

data = []
with open(csv_file, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        row['score'] = float(row['score'])
        data.append(row)

with open(json_file, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False)

import json
from tqdm import tqdm
with open('/content/banking_qwen.json', 'r', encoding="utf-8") as f:
  data = json.load(f)
with open('/content/all_scores_b_qwen.json', 'r', encoding="utf-8") as f:
  qscores = json.load(f)
  qsts = {elem['source']: elem['score'] for elem in qscores}
with open('/content/all_scores_b.json', 'r', encoding="utf-8") as f:
  scores = json.load(f)
  sts = {elem['source']: elem['score'] for elem in scores}
# splits = ['train', 'val', 'test', 'oos_train', 'oos_val', 'oos_test']

# results = {}

for split in splits:
    for item in tqdm(data[split]):
      if sts[item['text']]<qsts[item['text']] and 'qwen_translation' in item:
        item['translation'] = item['qwen_translation']
      elif sts[item['text']]<qsts[item['text']] and 'qwen_translation' not in item:
        print(item)
      if 'qwen_translation' in item:
        del item['qwen_translation']
      item['score'] = max(sts[item['text']], qsts[item['text']])

with open('banking_qwen2.json', "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False)

for split in data:
  if data[split]:
    score = sum([elem['score'] for elem in data[split]])/len(data[split])
    print(split, score)

# train 0.4674989903029091
# test 0.48108035714285713
# oos_train 0.34094055232558135
# oos_test 0.34366808176100627