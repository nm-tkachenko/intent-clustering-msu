import json
from tqdm import tqdm
import numpy as np
import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support, accuracy_score


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

from sklearn.metrics.pairwise import euclidean_distances

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
  return {'InClust': inclust[0]/max(inclust[1], 1), 'InterClust': interclust[0]/max(interclust[1], 1)}

from sklearn import metrics

def compute_metrics(pred_labels, gold_labels, dists, data, gold_ARPF=None, gold_B2=None):
  n_clusters_ = len(set(pred_labels)) - (1 if -1 in pred_labels else 0)
  n_noise_ = list(pred_labels).count(-1)
  for label in set(pred_labels):
    if list(pred_labels).count(label)==1: # для BIRCH
      n_noise_ += 1
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

from sklearn.cluster import DBSCAN
# from sklearn.cluster import HDBSCAN
from sklearn.cluster import OPTICS
from sklearn.cluster import Birch

def apply_DBSCAN(embeddings, gold, eps, min_samples):
  clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(embeddings)
  return clustering.labels_

# apply_DBSCAN(embeddings=embeddings, gold=test_data, eps=0.4, min_samples=2)

def apply_HDBSCAN(embeddings, gold, min_cluster_size, cluster_selection_epsilon):
  hdb = HDBSCAN(copy=True, min_cluster_size=min_cluster_size, cluster_selection_epsilon=cluster_selection_epsilon, allow_single_cluster=True)
  hdb.fit(embeddings)
  return hdb.labels_

def apply_OPTICS(embeddings, gold, max_eps, min_samples):
  clustering = OPTICS(min_samples=min_samples, max_eps=max_eps).fit(embeddings)
  return clustering.labels_

def apply_BIRCH(embeddings, gold, threshold, branching_factor):
  brc = Birch(threshold=threshold, branching_factor=branching_factor, n_clusters=None)
  brc.fit(embeddings)
  clustering = brc.predict(embeddings)
  return clustering

with open('clinc_qwen2.json', 'r', encoding="utf-8") as f:
  data = json.load(f)

train_data = []
for split in ('train', 'val', 'oos_test'):
  start = len(train_data)
  train_data.extend([(start+id_, elem['translation'], elem['label']) for id_, elem in enumerate(data[split])])

test_data = []
for split in ('oos_train', 'oos_val', 'test'):
  start = len(train_data)+len(test_data)
  test_data.extend([(start+id_, elem['translation'], elem['label']) for id_, elem in enumerate(data[split])])

labels_true=[elem[2] for elem in test_data]
labels_codes = {l: x for x, l in enumerate(set(labels_true))}
labels_codes['oos'] = -1
gold_labels = [labels_codes[l] for l in labels_true]
gold_ARPF = []
for i in tqdm(range(len(test_data))):
  for j in range(i+1, len(test_data)):
    if test_data[i][2]==test_data[j][2] and test_data[i][2]!='oos':
        # if data[i][2]==data[j][2]:
      gold_ARPF.append(1)
    else:
      gold_ARPF.append(0)
gold_B2 = {}
for i in tqdm(range(len(test_data))):
  label = test_data[i][2] if test_data[i][2]!='oos' else 'oos'+str(i)
  gold_B2[label] = gold_B2.get(label, set())
  gold_B2[label].add(i)

stats = []
for model_func, m in zip((apply_frida, apply_bge, apply_e5, apply_labse, apply_rosberta),
                          ('FRIDA', 'BGE', 'E5', 'LaBSE', 'RoSBERTa')):
    # embeddings = model_func(train_data)
    with open(f'embeds {m}.json', 'r', encoding="utf-8") as f:
      embeddings = json.load(f)
    dists = euclidean_distances(embeddings, embeddings)
  
    # compute_metrics(pred_labels, gold_labels=, dists=dists, data=test_data, gold_ARPF=gold_ARPF, gold_B2=gold_B2)

    for eps_int in range(1, 10, 2):
        eps = eps_int/10
        for min_samples in range(1, 6, 2):
            pred_labels = apply_DBSCAN(embeddings, train_data, eps=eps, min_samples=min_samples)
            metrics_ = compute_metrics(pred_labels=pred_labels, gold_labels=gold_labels, dists=dists, data=test_data, gold_ARPF=gold_ARPF, gold_B2=gold_B2)
            result = {'model': m, 'alg': 'DBSCAN', 'hyperparams': f'eps={eps}, min_samples={min_samples}'} | metrics_
            stats.append(result)
            # break
        # break
    with open('enc_clustering_stats.json', "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False)

    # for min_cluster_size in range(2, 7, 2):
    #     for cluster_selection_epsilon_int in range(50, 90, 10):
    #         cluster_selection_epsilon = cluster_selection_epsilon_int/100
    #         pred_labels = apply_HDBSCAN(embeddings, train_data, min_cluster_size=min_cluster_size, cluster_selection_epsilon=cluster_selection_epsilon)
    #         metrics_ = compute_metrics(pred_labels=pred_labels, gold_labels=gold_labels, dists=dists, data=test_data, gold_ARPF=gold_ARPF, gold_B2=gold_B2)
    #         result = {'model': m, 'alg': 'HDBSCAN', 
    #                       'hyperparams': f'min_cluster_size={min_cluster_size}, cluster_selection_epsilon={cluster_selection_epsilon}'} | metrics_
    #         stats.append(result)
    #         break
    #     break
    # with open('enc_clustering_stats.json', "w", encoding="utf-8") as f:
    #     json.dump(stats, f, ensure_ascii=False)
    
    for max_eps_int in range(5, 15, 2):
        max_eps = max_eps_int/10
        for min_samples in range(2, 7, 2):
            pred_labels = apply_OPTICS(embeddings, train_data, max_eps=max_eps, min_samples=min_samples)
            metrics_ = compute_metrics(pred_labels=pred_labels, gold_labels=gold_labels, dists=dists, data=test_data, gold_ARPF=gold_ARPF, gold_B2=gold_B2)
            result = {'model': m, 'alg': 'OPTICS', 
                          'hyperparams': f'max_eps={max_eps}, min_samples={min_samples}'} | metrics_
            stats.append(result)
            # break
        # break
    with open('enc_clustering_stats.json', "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False)

    for threshold_int in range(60, 90, 10):
        threshold = threshold_int/100
        for branching_factor in range(10, 70, 20):
            pred_labels = apply_BIRCH(embeddings, train_data, threshold=threshold, branching_factor=branching_factor)
            metrics_ = compute_metrics(pred_labels=pred_labels, gold_labels=gold_labels, dists=dists, data=test_data, gold_ARPF=gold_ARPF, gold_B2=gold_B2)
            result = {'model': m, 'alg': 'BIRCH', 
                          'hyperparams': f'threshold={threshold}, branching_factor={branching_factor}'} | metrics_
            stats.append(result)
            # break
        # break
    with open('enc_clustering_stats.json', "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False)