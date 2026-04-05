'''
This file contains all the functions that apply encoders to data and return embeddings.
'''
from sentence_transformers import SentenceTransformer

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

    