import json
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support, accuracy_score
from sklearn.metrics.pairwise import euclidean_distances
from sklearn import metrics
from sklearn.cluster import DBSCAN
from sklearn.cluster import Birch
from sentence_transformers import SentenceTransformer
import nltk
from rake_nltk import Rake
nltk.download('punkt_tab')

def apply_frida(data, prefix="paraphrase: ", convert_to_tensor=True):
  inputs = [prefix + elem[1] for elem in data]
  model = SentenceTransformer("ai-forever/FRIDA")
  embeddings = model.encode(inputs, convert_to_tensor=convert_to_tensor)
  return embeddings.cpu()

def apply_bge(data, prefix='', normalize_embeddings=True):
  inputs = [prefix + elem[1] for elem in data]
  model = SentenceTransformer("deepvk/USER-bge-m3")
  embeddings = model.encode(inputs, normalize_embeddings=normalize_embeddings)
  return embeddings

def APRF_pairwise(data, result, gold=None):
  if gold is None:
    gold = []
    for i in tqdm(range(len(data))):
      for j in range(i+1, len(data)):
        if data[i][2]==data[j][2] and data[i][2]!='oos':
        # if data[i][2]==data[j][2]:
          gold.append(1)
        else:
          gold.append(0)
  pred = []
  for i in tqdm(range(len(result))):
    for j in range(i+1, len(result)):
      if result[i]==result[j]:
        pred.append(1)
      else:
        pred.append(0)
  # F1 = f1_score(y_true, y_pred, average='macro')
  P, R, F1, s = precision_recall_fscore_support(gold, pred, average='macro')
  Acc = accuracy_score(gold, pred)
  return {"Accuracy": Acc, "P": P, "R": R, "F1": F1}

def b_cubed(data, result, gold=None):
  N = len(data)
  assert N == len(result)
  if gold is None:
    gold = {}
    for i in tqdm(range(N)):
      label = data[i][2] if data[i][2]!='oos' else 'oos'+str(i)
      gold[label] = gold.get(label, set())
      gold[label].add(i)
  pred = {}
  for i in tqdm(range(N)):
    label = result[i]
    pred[label] = pred.get(label, set())
    pred[label].add(i)
  BCP_sum, BCR_sum = 0, 0
  oos_sum, oos_N, oos_count = 0, 0, 0
  for i in tqdm(range(N)):
    t_d = gold[data[i][2]] if data[i][2]!='oos' else gold['oos'+str(i)]
    c_d = pred[result[i]]
    inter = len(t_d.intersection(c_d))
    BCP_sum += inter/len(c_d)
    BCR_sum += inter/len(t_d)
    if data[i][2]=='oos':
      oos_sum += 1/len(c_d)
      oos_count += int(len(c_d)==1)
      oos_N += 1
  BCP, BCR = BCP_sum/N, BCR_sum/N
  BCF = 2*BCP*BCR/(BCP+BCR)
  oos_detection = oos_sum/oos_N if oos_N else 'undefined'
  oos_percent = oos_count/oos_N if oos_N else 'undefined'
  return {'BCP': BCP, 'BCR': BCR, 'BCF': BCF, 'oos_detection': oos_detection, 'oos_percent': oos_percent}

def mean_in_inter(result, dists):
  N = len(result)
  inclust = [0, 0]
  interclust = [0, 0]
  for i in tqdm(range(N)):
    for j in range(i, N):
      if result[i] == result[j]:
        inclust[0] += dists[i][j]
        inclust[1] += 1
      else:
        interclust[0] += dists[i][j]
        interclust[1] += 1
  return {'InClust': float(inclust[0]/max(inclust[1], 1)), 'InterClust': float(interclust[0]/max(interclust[1], 1))}

def compute_metrics(pred_labels, gold_labels, dists, data, gold_ARPF=None, gold_B2=None):
  n_clusters_ = len(set(pred_labels)) - (1 if -1 in pred_labels else 0)
  n_noise_ = list(pred_labels).count(-1)
  for label in set(pred_labels):
    if list(pred_labels).count(label)==1: # для BIRCH
      n_noise_ += 1
      n_clusters_ -= 1
  try:
    SC = metrics.silhouette_score(dists, pred_labels)
  except ValueError:
    SC = -1
  rezult = {'N_clusters': n_clusters_, 'N_noise': n_noise_,
  'Homogeneity': metrics.homogeneity_score(gold_labels, pred_labels),
  'Completeness': metrics.completeness_score(gold_labels, pred_labels),
  'V-measure': metrics.v_measure_score(gold_labels, pred_labels),
  'Adjusted Rand Index': metrics.adjusted_rand_score(gold_labels, pred_labels),
  'Adjusted Mutual Information': metrics.adjusted_mutual_info_score(gold_labels, pred_labels),
  'Silhouette Coefficient': SC} 
  rezult = rezult | APRF_pairwise(data, pred_labels, gold=gold_ARPF) | b_cubed(data, pred_labels, gold=gold_B2) | mean_in_inter(pred_labels, dists)
  return rezult

def keywords(labels, data):
    pred = {}
    for i, label in tqdm(enumerate(labels)):
        if label!=-1:
          pred[label] = pred.get(label, [])
          pred[label].append(data[i][1])
    corpus = [{'support': len(pred[label]), 'text': '\n'.join(pred[label])} for label in pred]
    r = Rake(min_length=2, max_length=6)
    for text in corpus:
      r.extract_keywords_from_text(text["text"])
      keywords = r.get_ranked_phrases()
      text["rake_results"] = keywords
    return sorted(corpus, key=lambda elem: -elem['support'])

def apply_DBSCAN(embeddings, eps, min_samples):
  clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(embeddings)
  return clustering.labels_

def apply_BIRCH(embeddings, threshold, branching_factor):
  brc = Birch(threshold=threshold, branching_factor=branching_factor, n_clusters=None)
  brc.fit(embeddings)
  clustering = brc.predict(embeddings)
  return clustering

def execute(data, model_func=apply_bge, clustering_method='BIRCH', 
            threshold=0.55, branching_factor=30, eps=0.5, min_samples=1):
    labels_true=[elem[2] for elem in data]
    labels_codes = {l: x for x, l in enumerate(set(labels_true))}
    labels_codes['oos'] = -1
    gold_labels = [labels_codes[l] for l in labels_true]
    gold_ARPF = []
    for i in range(len(data)):
        for j in range(i+1, len(data)):
            if data[i][2]==data[j][2] and data[i][2]!='oos':
                # if data[i][2]==data[j][2]:
                gold_ARPF.append(1)
            else:
                gold_ARPF.append(0)
    gold_B2 = {}
    for i in range(len(data)):
        label = data[i][2] if data[i][2]!='oos' else 'oos'+str(i)
        gold_B2[label] = gold_B2.get(label, set())
        gold_B2[label].add(i)
    embeddings = model_func(data)
    dists = euclidean_distances(embeddings, embeddings)
    if clustering_method=='BIRCH':
        pred_labels = apply_BIRCH(embeddings, threshold=threshold, branching_factor=branching_factor)
    elif clustering_method=='DBSCAN':
        pred_labels = apply_DBSCAN(embeddings, eps=eps, min_samples=min_samples)
    else:
        print('unsupported clustering method')
    metrics_ = compute_metrics(pred_labels=pred_labels, gold_labels=gold_labels, dists=dists, data=data, gold_ARPF=gold_ARPF, gold_B2=gold_B2)
    return {'metrics': metrics_, 'clusters and keywords': keywords(pred_labels, data), 'pred_labels': pred_labels.tolist()}
            
with open('clinc_qwen2.json', 'r', encoding="utf-8") as f:
  data = json.load(f)

train_data = []
for split in ('train', 'val', 'oos_test'):
# for split in ('val',):
  start = len(train_data)
  train_data.extend([(start+id_, elem['translation'], elem['label']) for id_, elem in enumerate(data[split])])

rezult = execute(train_data)
# rezult = execute(train_data, model_func=apply_frida, clustering_method='DBSCAN')
print(rezult['metrics'])
with open('rezult_.json', "w", encoding="utf-8") as f:
    json.dump(rezult, f, ensure_ascii=False)