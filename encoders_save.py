import json
from tqdm import tqdm
import numpy as np

from sentence_transformers import SentenceTransformer

def apply_frida(data, prefix="paraphrase: ", convert_to_tensor=True):
  inputs = [prefix + elem[1] for elem in data]
  model = SentenceTransformer("ai-forever/FRIDA")
  embeddings = model.encode(inputs, cgit@github.com:nm-tkachenko/intent-clustering-msu.gitonvert_to_tensor=convert_to_tensor)
  return embeddings.cpu()

frida_prefixes = ["search_query: ", "search_document: ",# prefixes are for answer or relevant paragraph retrieval
                  "paraphrase: ",# prefix is for symmetric paraphrasing related tasks (STS, paraphrase mining, deduplication)
                  "categorize: ",# prefix is for asymmetric matching of document title and body (e.g. news, scientific papers, social posts)
                  "categorize_sentiment: ",# prefix is for any tasks that rely on sentiment features (e.g. hate, toxic, emotion)
                  "categorize_topic: ",# prefix is intended for tasks where you need to group texts by topic
                  "categorize_entailment: "]# prefix is for textual entailment task (NLI)

def apply_bge(data, prefix='', normalize_embeddings=True):
  inputs = [prefix + elem[1] for elem in data]
  model = SentenceTransformer("deepvk/USER-bge-m3")
  embeddings = model.encode(inputs, normalize_embeddings=normalize_embeddings)
  return embeddings

def apply_e5(data, prefix='', convert_to_tensor=True, normalize_embeddings=False):
  inputs = [prefix + elem[1] for elem in data]
  model = SentenceTransformer('intfloat/multilingual-e5-large-instruct')
  embeddings = model.encode(inputs, convert_to_tensor=convert_to_tensor, normalize_embeddings=normalize_embeddings)
  return embeddings

def apply_labse(data, prefix=''):
  inputs = [prefix + elem[1] for elem in data]
  model = SentenceTransformer('sentence-transformers/LaBSE')
  embeddings = model.encode(inputs)
  return embeddings

def apply_rosberta(data, prefix="classification: ", convert_to_tensor=True):
  inputs = [prefix + elem[1] for elem in data]
  model = SentenceTransformer("ai-forever/ru-en-RoSBERTa")
  embeddings = model.encode(inputs, convert_to_tensor=convert_to_tensor)
  return embeddings

rosberta_prefixes = ["classification: ",
                     "clustering: ",
                     "search_query: ",
                     "search_document: "]


with open('clinc_qwen2.json', 'r', encoding="utf-8") as f:
  data = json.load(f)

train_data = []
for split in ('train', 'val', 'oos_test'):
  start = len(train_data)
  train_data.extend([(start+id, elem['translation'], elem['label']) for id, elem in enumerate(data[split])])

test_data = []
for split in ('oos_train', 'oos_val', 'test'):
  start = len(train_data)+len(test_data)
  test_data.extend([(start+id, elem['translation'], elem['label']) for id, elem in enumerate(data[split])])

for model_func, m in zip((apply_frida, apply_bge, apply_e5, apply_labse, apply_rosberta),
                          ('FRIDA', 'BGE', 'E5', 'LaBSE', 'RoSBERTa')):
    embeddings = model_func(test_data)

    try:
        with open(f'embeds {m}.json', "w", encoding="utf-8") as f:
            json.dump(embeddings.cpu().tolist(), f, ensure_ascii=False)
    except:
        print(m)
        with open(f'embeds {m}.json', "w", encoding="utf-8") as f:
            json.dump(embeddings.tolist(), f, ensure_ascii=False)

    