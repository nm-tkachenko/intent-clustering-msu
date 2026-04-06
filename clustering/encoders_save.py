import json
from functions.encoders import *

'''
This code applies all the encoders on train data and save the embeddings.
'''

with open('../translation/clinc_qwen2.json', 'r', encoding="utf-8") as f:
  data = json.load(f)

train_data = []
for split in ('oos_train', 'oos_val', 'test'):
  start = len(train_data)
  train_data.extend([(start+id, elem['translation'], elem['label']) for id, elem in enumerate(data[split])])

# test_data = []
# for split in ('train', 'val', 'oos_test'):
#   start = len(train_data)+len(test_data)
#   test_data.extend([(start+id, elem['translation'], elem['label']) for id, elem in enumerate(data[split])])

for model_func, m in zip((apply_frida, apply_bge, apply_e5, apply_labse, apply_rosberta),
                          ('FRIDA', 'BGE', 'E5', 'LaBSE', 'RoSBERTa')):
    embeddings = model_func(train_data)

    try:
        with open(f'embeddings/clinc_{m}.json', "w", encoding="utf-8") as f:
            json.dump(embeddings.cpu().tolist(), f, ensure_ascii=False)
    except:
        # print(m)
        with open(f'embeddings/clinc_{m}.json', "w", encoding="utf-8") as f:
            json.dump(embeddings.tolist(), f, ensure_ascii=False)

with open('../translation/banking_qwen2.json', 'r', encoding="utf-8") as f:
  data = json.load(f)

train_data = []
for split in ('test', 'val', 'oos_test', 'oos_val'):
  start = len(train_data)
  train_data.extend([(start+id, elem['translation'], elem['label']) for id, elem in enumerate(data[split])])

# test_data = []
# for split in ('train', 'oos_train'):
#   start = len(train_data)+len(test_data)
#   test_data.extend([(start+id, elem['translation'], elem['label']) for id, elem in enumerate(data[split])])

for model_func, m in zip((apply_frida, apply_bge, apply_e5, apply_labse, apply_rosberta),
                          ('FRIDA', 'BGE', 'E5', 'LaBSE', 'RoSBERTa')):
    embeddings = model_func(train_data)

    try:
        with open(f'embeddings/banking_{m}.json', "w", encoding="utf-8") as f:
            json.dump(embeddings.cpu().tolist(), f, ensure_ascii=False)
    except:
        # print(m)
        with open(f'embeddings/banking_{m}.json', "w", encoding="utf-8") as f:
            json.dump(embeddings.tolist(), f, ensure_ascii=False)
    