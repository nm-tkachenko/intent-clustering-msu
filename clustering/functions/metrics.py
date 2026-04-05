'''
This file contains all the functions used for evaluation.
'''

from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.metrics.pairwise import euclidean_distances
from sklearn import metrics
import nltk
from rake_nltk import Rake
nltk.download('punkt_tab')

def APRF_pairwise(data, result, gold=None):
  if gold is None:
    gold = []
    for i in tqdm(range(len(data))):
      for j in range(i+1, len(data)):
        if data[i][2]==data[j][2] and data[i][2]!='oos':
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
    if list(pred_labels).count(label)==1:
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