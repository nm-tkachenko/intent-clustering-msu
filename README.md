# Intent Extraction and Clustering for Contact Center Analytics

Intent extraction from user-assistant conversations and unsupervised intent clustering using Encoder models.

Our goal is to create a cost-efficient pipeline for automated grouping and describing users' intents from conversations with a chatbot assistant. The pipeline could be used by customer service for early detecting of new topics and refining the algorithms for better user experience.

This project is split into 3 following steps:
1. Automated data translation (from English to Russian)
2. Experiments with different models and clustering methods
3. Cluster names generation (intent extraction)

## Data translation
We use the following open-source datasets for the hyperparameter tuning and evaluation:
* **CLINC150**: queries from diverse domains annotated with their corresponding intents mixed with out-of-scope intent examples.

* **banking-77**: online banking queries annotated with their corresponding intents.

  This dataset doesn't contain any out-of-scope classes relevant for our task, so we included additional data from Natural Questions dataset (2700 examples, ~20% of original banking-77 dataset size).

As the base for translation from English to Russian we used open-source model OmniLing-V1-8b (8-bit) post-processed with the regular expressions. Translation quality was evaluated with COMET metric: low-quality examples (score < 0.2) were regenerated with RuadaptQwen3-8B-Hybrid. For pairs of original OmniLing and additional RuadaptQwen3 translations, the variant with the best metric score was included in the final data.

Files containing the final translations can be found in the *Translation* folder of the project.

## Clustering
We experiment with five models for clustering: **FRIDA**, **E5**, **BGE**, **LaBSE** and **RoSBERTa**. Each of the models was tested with different clustering methods to discover the best pipeline structure and further improve the results by more precise fine-tuning.

Three clustering methods were implemented in this project: **DBSCAN**, **OPTICS** and **BIRCH**. These methods were chosen for their effectiveness for dealing with large multi-cluster data with no fixed cluster count and outlier removal.

Important note: for balanced data split this project divides original subsets of the data as follows:
* *train* data of this project consists of `oos_train`, `oos_val`, `test` dataset splits due to the smaller size.
* *test* data of this project consists of `train`, `val`, `oos_test` dataset splits to avoid overfitting and unrepresentative results.

For each method, two key hyperparameters were fine-tuned on the train data. Two best pipelines were chosen for test evaluation. 
| Method | Hyperparameters |
|---------|--------|
| DBSCAN | `eps`, `min_samples` |
| OPTICS | `max_eps`, `min_samples` |
| BIRCH | `threshold`, `branching_factor` |

Visualization of clustering parameters finetuning and corresponding metric scores canbe found in the *Visualization* folder of the project.

Clustering quality was evaluated with 17 different metrics. Exact list as well as the scores for the best methods can be found in the *Clustering* folder of the project. The main 6 metrics are represented in the *Results* section.

## Intent extraction

Intent for each discovered cluster (cluster name) was generated with the **RAKE** algorithm. Keyword phrases with the highest scores were attributed to the corresponding cluster elements. List of clusters and the corresponding intents (sorted by RAKE score from highest to lowest) can be found in the *Clustering* folder of the project.

TBA: improving the pipeline for this subtask, intent extraction quality evaluation.

## Results
The two best pipelines for **CLINC150** as follows:
| Split | Model | Method | Hyperparameters | Number of clusters | Noise clusters |
|---------|--------|--------|--------|--------|--------|
| train | BGE | BIRCH | `threshold=0.55`, `branching_factor=30` | 1099 | 604 |
| train | FRIDA | DBSCAN | `eps=0.5`, `min_samples=1` | 1895 | 1535 |
| test | BGE | BIRCH | `threshold=0.55`, `branching_factor=30` | 1516 | 2077 |
| test | FRIDA | DBSCAN | `eps=0.5`, `min_samples=1` | 707	| 4574 |

| Split | Model | Method | Hyperparameters | V-measure | Adjusted Mutual Information | Silhouette Coefficient | F1 | BCF | oos_detection |
|---------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| train | BGE | BIRCH | `threshold=0.55`, `branching_factor=30` | 0.84 | 0.69 | 0.00063 | 0.76 | 0.60 | 0.69 |
| train | FRIDA | DBSCAN | `eps=0.5`, `min_samples=1` | 0.82 | 0.60 | 0.015 | 0.72 | 0.55	| 0.97 |
| test | BGE | BIRCH | `threshold=0.55`, `branching_factor=30` | 0.79	| 0.66 | -0.06	| 0.71	| 0.49	| 0.66 |
| test | FRIDA | DBSCAN | `eps=0.5`, `min_samples=1` | 0.73	| 0.58 | -0.23 |	0.55	| 0.59	| 0.94 |

The two best pipelines for **banking-77** as follows:
| Split | Model | Method | Hyperparameters | Number of clusters | Noise clusters |
|---------|--------|--------|--------|--------|--------|
| train | FRIDA | BIRCH | `threshold=0.5`, `branching_factor=40` | 256 | 456 |
| train | BGE | BIRCH | `threshold=0.5`, `branching_factor=40` | 285 | 589 |
| test | FRIDA | BIRCH | `threshold=0.5`, `branching_factor=40` | 578 | 1285 |
| test | BGE | BIRCH | `threshold=0.5`, `branching_factor=40` | 726	| 1537 |

| Split | Model | Method | Hyperparameters | V-measure | Adjusted Mutual Information | Silhouette Coefficient | F1 | BCF | oos_detection |
|---------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| train | BGE | BIRCH | `threshold=0.5`, `branching_factor=40` | 0.72	| 0.57 |	-0.0074	| 0.72	| 0.59	| 0.72 |
| train | BGE | BIRCH | `threshold=0.5`, `branching_factor=40` | 0.72	| 0.55	| -0.037	| 0.73	| 0.59	| 0.85 |
| test | FRIDA | BIRCH | `threshold=0.5`, `branching_factor=40` | 0.70	| 0.60	| -0.027 |	0.73	| 0.58	| 0.70 |
| test | BGE | BIRCH | `threshold=0.5`, `branching_factor=40` | 0.69	| 0.57 |	-0.055	| 0.70 |	0.54 |	0.81 |

---
This project was created for the academic course "Project tasks in Computer Linguistics" (MSU, 2026).

Authors: Lapanitsyna Anna, Tkachenko Natalia, Bolotova Maria
