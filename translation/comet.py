'''
This code uses wmt20-comet-qe-da to evaluate OmniLing-V1-8b translations.
'''

from comet import download_model, load_from_checkpoint

model_path = download_model("Unbabel/wmt20-comet-qe-da")
model = load_from_checkpoint(model_path)

import csv
import json
with open('/content/clinc_handcheck.json', 'r', encoding="utf-8") as f:
# with open('/content/banking77_handcheck.json', 'r', encoding="utf-8") as f:
  data = json.load(f)
splits = ['train', 'val', 'test', 'oos_train', 'oos_val', 'oos_test']
results = {}

with open('all_scores.csv', 'w', newline='', encoding='utf-8') as f_csv:
# with open('all_scores_b.csv', 'w', newline='', encoding='utf-8') as f_csv:
    writer = csv.writer(f_csv)
    writer.writerow(['split', 'source', 'translation', 'score'])

    for split in splits:
        if split not in data:
            continue

        items = data[split]
        if not items:
            continue

        data_for_model = []
        for item in items:
            src = item.get('text')
            mt = item.get('translation')
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

csv_file = 'all_scores.csv'
# csv_file = 'all_scores_b.csv'
json_file = 'all_scores.json'
# json_file = 'all_scores_b.json'

data = []
with open(csv_file, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        row['score'] = float(row['score'])
        data.append(row)

with open(json_file, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False)