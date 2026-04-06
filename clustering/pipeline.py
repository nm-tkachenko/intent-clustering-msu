import json
from tqdm import tqdm
from functions.metrics import *
from functions.algorithms import *
from functions.encoders import *

'''
This file executes the best combination of clustering algorithm, its parameters and embeddings used on the test sets of both datasets,
evaluates the results and saves the statistics (RAKE results included).
This is also an example of using the entire repository.
'''

def execute(data, model_func=apply_bge, clustering_method='BIRCH', 
            threshold=0.5, branching_factor=30, eps=0.5, min_samples=1):
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

for ds in ('clinc', 'banking'):
    if ds=='clinc':
      continue
  
    with open(f'../translation/{ds}_qwen2.json', 'r', encoding="utf-8") as f:
      data = json.load(f)

    # train_data = []
    # train_splits = ('oos_train', 'oos_val', 'test') if ds=='clinc' else ('test', 'val', 'oos_test', 'oos_val')
    # for split in train_splits:
    #   start = len(train_data)
    #   train_data.extend([(start+id, elem['translation'], elem['label']) for id, elem in enumerate(data[split])])
    
    test_data = []
    test_splits = ('train', 'val', 'oos_test') if ds=='clinc' else ('train', 'oos_train')
    for split in test_splits:
       start = len(test_data)# + len(train_data)
       test_data.extend([(start+id, elem['translation'], elem['label']) for id, elem in enumerate(data[split])])
    
    if ds=='clinc':
        result = execute(test_data, model_func=apply_bge, clustering_method='BIRCH', threshold=0.55, branching_factor=30)
        # result = execute(test_data, model_func=apply_frida, clustering_method='DBSCAN', eps=0.5, min_samples=1)
        with open(f'result_bge+birch_{ds}.json', "w", encoding="utf-8") as f:
      # with open(f'result_frida+dbscan_{ds}.json', "w", encoding="utf-8") as f:
          json.dump(result, f, ensure_ascii=False)
    elif ds=='banking':
        result = execute(test_data, model_func=apply_bge, clustering_method='BIRCH', threshold=0.5, branching_factor=40)
        # result = execute(test_data, model_func=apply_frida, clustering_method='BIRCH', threshold=0.5, branching_factor=40)
        with open(f'result_bge+birch_{ds}.json', "w", encoding="utf-8") as f:
        # with open(f'result_frida+birch_{ds}.json', "w", encoding="utf-8") as f:
          json.dump(result, f, ensure_ascii=False)     
    print(result['metrics'])