import json
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('punkt')
stop_words = list(set(stopwords.words('russian')))

def tf_idf_keywords(data, ngram_range=(2,3), min_df=0.15):
  vectorizer = TfidfVectorizer(ngram_range=ngram_range, stop_words=stop_words, min_df=min_df)
  try:
    vectorizer.fit_transform(data)
    return vectorizer.get_feature_names_out()
  except ValueError:
    pass

with open(f'result_bge+birch_clinc.json', 'r', encoding="utf-8") as f:
    data = json.load(f)

for cluster in data['clusters and keywords']:
    print(cluster['support'])
    texts = cluster['text'].split('\n')
    print('\n'.join(texts[:10]))
    print(cluster['rake_results'][:5])
    print(tf_idf_keywords(texts))
    if input()=='break':
       break

with open(f'result_bge+birch_banking.json', 'r', encoding="utf-8") as f:
    data = json.load(f)

for cluster in data['clusters and keywords']:
    print(cluster['support'])
    texts = cluster['text'].split('\n')
    print('\n'.join(texts[:10]))
    print(cluster['rake_results'][:5])
    print(tf_idf_keywords(texts))
    if input()=='break':
       break