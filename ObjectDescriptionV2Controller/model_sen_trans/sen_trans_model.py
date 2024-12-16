from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
from FlagEmbedding import BGEM3FlagModel

sentences = ["man wearing shoe, sidewalk near street, person wearing shirt, man riding skateboard, pant on man, man has head, man wearing jean, man has hair"]

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embeddings = model.encode(sentences)
embeddings = torch.tensor(embeddings)
print(embeddings.size())

model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
embeddings = model.encode(sentences)
embeddings = torch.tensor(embeddings)
print(embeddings.size())

model = BGEM3FlagModel('BAAI/bge-m3',
       use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation
embeddings = model.encode(sentences,
        batch_size=12,
        max_length=8192, # If you don't need such a long length, you can set a smaller value to speed up the encoding process.
        )['dense_vecs']

embeddings=torch.tensor(embeddings)
print(embeddings.size())