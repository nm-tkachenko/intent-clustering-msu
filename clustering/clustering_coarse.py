import json
from tqdm import tqdm
from functions.metrics import *
from functions.algorithms import *

'''
This code runs all the clustering algorithms on the training sets of both datasets encoded by all the models, 
roughly iterating over the parameters, evaluates the results and saves the statistics (RAKE results excluded). 
'''

for ds in ('clinc', 'banking'):
    if ds=='clinc':
      continue
  
    with open(f'../translation/{ds}_qwen2.json', 'r', encoding="utf-8") as f:
      data = json.load(f)

    train_data = []
    train_splits = ('oos_train', 'oos_val', 'test') if ds=='clinc' else ('test', 'val', 'oos_test', 'oos_val')
    for split in train_splits:
      start = len(train_data)
      train_data.extend([(start+id, elem['translation'], elem['label']) for id, elem in enumerate(data[split])])

    labels_true=[elem[2] for elem in train_data]
    labels_codes = {l: x for x, l in enumerate(set(labels_true))}
    labels_codes['oos'] = -1
    gold_labels = [labels_codes[l] for l in labels_true]
    gold_ARPF = []
    for i in tqdm(range(len(train_data))):
      for j in range(i+1, len(train_data)):
        if train_data[i][2]==train_data[j][2] and train_data[i][2]!='oos':
          gold_ARPF.append(1)
        else:
          gold_ARPF.append(0)
    gold_B2 = {}
    for i in tqdm(range(len(train_data))):
      label = train_data[i][2] if train_data[i][2]!='oos' else 'oos'+str(i)
      gold_B2[label] = gold_B2.get(label, set())
      gold_B2[label].add(i)

    stats = []
    for m in ('FRIDA', 'BGE', 'E5', 'LaBSE', 'RoSBERTa'):
        with open(f'embeddings/{ds}_{m}.json', 'r', encoding="utf-8") as f:
          embeddings = json.load(f)
        dists = euclidean_distances(embeddings, embeddings)
      
        for eps_int in range(1, 10, 2):
            eps = eps_int/10
            for min_samples in range(1, 6, 2):
                pred_labels = apply_DBSCAN(embeddings, eps=eps, min_samples=min_samples)
                metrics_ = compute_metrics(pred_labels=pred_labels, gold_labels=gold_labels, dists=dists, data=train_data, gold_ARPF=gold_ARPF, gold_B2=gold_B2)
                result = {'model': m, 'alg': 'DBSCAN', 'hyperparams': f'eps={eps}, min_samples={min_samples}'} | metrics_
                stats.append(result)
                # break
            # break
        with open(f'clustering_coarse_{ds}.json', "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False)

        # for min_cluster_size in range(2, 7, 2):
        #     for cluster_selection_epsilon_int in range(50, 90, 10):
        #         cluster_selection_epsilon = cluster_selection_epsilon_int/100
        #         pred_labels = apply_HDBSCAN(embeddings, min_cluster_size=min_cluster_size, cluster_selection_epsilon=cluster_selection_epsilon)
        #         metrics_ = compute_metrics(pred_labels=pred_labels, gold_labels=gold_labels, dists=dists, data=train_data, gold_ARPF=gold_ARPF, gold_B2=gold_B2)
        #         result = {'model': m, 'alg': 'HDBSCAN', 
        #                       'hyperparams': f'min_cluster_size={min_cluster_size}, cluster_selection_epsilon={cluster_selection_epsilon}'} | metrics_
        #         stats.append(result)
        #         break
        #     break
        # with open(f'clustering_coarse_{ds}.json', "w", encoding="utf-8") as f:
        #     json.dump(stats, f, ensure_ascii=False)
        
        for max_eps_int in range(5, 15, 2):
            max_eps = max_eps_int/10
            for min_samples in range(2, 7, 2):
                pred_labels = apply_OPTICS(embeddings, max_eps=max_eps, min_samples=min_samples)
                metrics_ = compute_metrics(pred_labels=pred_labels, gold_labels=gold_labels, dists=dists, data=train_data, gold_ARPF=gold_ARPF, gold_B2=gold_B2)
                result = {'model': m, 'alg': 'OPTICS', 
                              'hyperparams': f'max_eps={max_eps}, min_samples={min_samples}'} | metrics_
                stats.append(result)
                # break
            # break
        with open(f'clustering_coarse_{ds}.json', "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False)

        for threshold_int in range(60, 90, 10):
            threshold = threshold_int/100
            for branching_factor in range(10, 70, 20):
                pred_labels = apply_BIRCH(embeddings, threshold=threshold, branching_factor=branching_factor)
                metrics_ = compute_metrics(pred_labels=pred_labels, gold_labels=gold_labels, dists=dists, data=train_data, gold_ARPF=gold_ARPF, gold_B2=gold_B2)
                result = {'model': m, 'alg': 'BIRCH', 
                              'hyperparams': f'threshold={threshold}, branching_factor={branching_factor}'} | metrics_
                stats.append(result)
                # break
            # break
        with open(f'clustering_coarse_{ds}.json', "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False)