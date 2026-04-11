# Clustering folder structure

This folder contains code and data relevant to the 2 of the project subtasks: **clustering** and **intent extraction**.

The main file of this folder is `pipeline.py`: this file executes the best combination of model, clustering algorithm, its parameters and embeddings used on the test sets of both datasets,
evaluates the results and saves the statistics (RAKE results included).

This is the pipeline of the project functioning as an example of using the entire repository. Other relevant files presented in the repository are described below.

## Code files
Files presented in the main folder space:
* `encoders_save.py`: code for applying all encoder models to data and returning embeddings created by corresponding models.
* `clustering_coarse.py`: code for performing rough evaluation on the train data and further pipeline selection.
* `clustering_coarse_added.py`: auxiliary code for rough evaluation with additional iterated hyperparameters.
* `clustering_fine.py`: code for performing hyperparameter finetuning and evaluation on train data using the most promising pipelines from the previous step.
* `keywords.py`: code for extracting keywords from a cluster by using TF-IDF vectorizer. This pipeline for intent extraction is currently WIP and will be expanded later.

Files presented in the *functions* subfolder contain the lists of all functions involved in certain steps of data processing:
* `algorithms.py`: all the functions used for performing clustering algorithms on a list of embeddings.
* `encoders.py`: all the functions used for applying encoders to data and returning embeddings.
* `metrics.py`: all the functions used for evaluation (metric scores computation).

## Clustering data files
In the following two files selections filename represents the clustering pipeline used for its creation (check *Code files* section for details).

Important note: for balanced data split this project divides original subsets of the data as follows:
* *train* data of this project consists of `oos_train`, `oos_val`, `test` dataset splits.
* *test* data of this project consists of `train`, `val`, `oos_test` dataset splits.

Clustered train data for rough evaluation contains a list of model, algorithm and hyperparameters combinations paired with corresponding metric scores:
* `clustering_coarse_clinc.json`
* `clustering_coarse_clinc_addition.json`
* `clustering_coarse_banking.json`
* `clustering_coarse_banking_addition.json`

Clustered train data for fine evaluation contains a list of model, algorithm and hyperparameters combinations paired with corresponding metric scores and cluster structure (elements of the cluster and extracted intent (RAKE results)):
* `clustering_fine_clinc.json`
* `clustering_fine_banking.json`

Clustered test data contains clustering results for the two best pipelines: extracted clusters with corresponding elements and RAKE-generated intents, predicted classes for each element of the test data and clustering metric scores.

## Results files
The table `all_metrics_clustering.tsv` contains the full list of metric scores used for evaluation of CLINC150 and banking-77 datasets. It represents two best pipelines for each dataset as well as results for both train and test data split.
