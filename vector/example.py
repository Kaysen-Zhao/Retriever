import sys, os
sys.path.append('vector/contriever')
sys.path.append(os.getcwd())

from src.contriever import Contriever
from transformers import AutoTokenizer
from utiles import ConfigFileHandler

config = ConfigFileHandler()

method = 'contriever'

path = config.get(['vector', 'methods', method, 'model_path', 'question_encoder'])
path_ref = config.get(['vector', 'methods', method, 'model_path', 'reference_encoder'])

device = config.get(['vector', 'device'])  # cuda cpu
model = Contriever.from_pretrained(path).to(device)
model_ref = Contriever.from_pretrained(path_ref).to(device)
tokenizer = AutoTokenizer.from_pretrained(path) #Load the associated tokenizer:


sentences = [
    "居里夫人的生日?",
]
sentences_ref = [
    "Where was Marie Curie born?",
    "Maria Sklodowska, later known as Marie Curie, was born on November 7, 1867.",
    "Born in Paris on 15 May 1859, Pierre Curie was the son of Eugène Curie, a doctor of French Catholic origin from Alsace."
]

inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt").to(device)
embeddings = model(**inputs)

inputs = tokenizer(sentences_ref, padding=True, truncation=True, return_tensors="pt").to(device)
embeddings_ref = model_ref(**inputs)

print(embeddings.shape)
for e in embeddings_ref:
    score01 = e @ embeddings[0]
    print(score01)
