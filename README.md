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

## Clustering
We experiment with five models for clustering: **FRIDA**, **E5**, **BGE**, **LaBSE** and **RoSBERTa**. Each of the models was tested with different clustering methods to discover the best pipeline structure and further improve the results by more precise fine-tuning.

Three clustering methods were implemented in this project: **DBSCAN**, **OPTICS** and **BIRCH**. These methods were chosen for their effectiveness for dealing with large multi-cluster data with no fixed cluster count and outlier removal.

For each method, two key hyperparameters were fine-tuned on the test split of the data due to the smaller size. Actual evaluation was done on the train split to avoid overfitting and unrepresentative results. 
| Method | Hyperparameters |
|---------|--------|
| DBSCAN | `eps`, `min_samples` |
| OPTICS | `max_eps`, `min_samples` |
| BIRCH | `threshold`, `branching_factor` |

Clustering quality was evaluated with 17 different metrics. Exact list as well as the scores for the best methods can be found in the *Results* section.

## Intent extraction

Intent for each discovered cluster (cluster name) was generated with the **RAKE** algorithm. Keyword phrases with the highest scores were attributed to the corresponding cluster elements.

## Results
The two best pipeline as follows:
| Model | Method | Hyperparameters | Number of clusters | Noise clusters | Homogeneity | Completeness | V-measure | 
|---------|--------|--------|--------|--------|--------|--------|--------|
| BGE | BIRCH | `threshold=0,55`, `branching_factor=30` | 3593 | 2077 | 0.94 | 0.68 | 0.79 |
| FRIDA | DBSCAN | `eps=0,5`, `min_samples=1` | 5281 | 4574 | 0.79 | 0.67 | 0.73 |

| Model | Method | Hyperparameters | Adjusted Rand Index | Adjusted Mutual Information | Silhouette Coefficient | Accuracy | P | R | F1 |
|---------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| BGE | BIRCH | `threshold=0,55`, `branching_factor=30` | 0.31 | 0.66 | -0.06 | 0.99 | 0.82 | 0.65 | 0.71 |
| FRIDA | DBSCAN | `eps=0,5`, `min_samples=1` | 0.096 | 0.58 | -0.23 | 0.95 | 0.53 | 0.73| 0.55 |

| Model | Method | Hyperparameters | BCP | BCR | BCF | oos_detection | oos_percent | InClust | InterClust |
|---------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| BGE | BIRCH | `threshold=0,55`, `branching_factor=30` | 0.83 | 0.35 | 0.49 | 0.66 | 0.51 | 0.65 | 0.19 |
| FRIDA | DBSCAN | `eps=0,5`, `min_samples=1` | 0.66 | 0.54 | 0.59 | 0.94 | 0.91 | 0.93 | 0.20 |

---
This project was created for the academic course "Project tasks in Computer Linguistics" (MSU, 2026).
Authors: Lapanitsyna Anna, Tkachenko Natalia, Bolotova Maria
