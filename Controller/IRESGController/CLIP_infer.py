import torch
from typing import Iterable
import time
import numpy as np
from tqdm import tqdm
import faiss
from collections import defaultdict
from PIL import Image, ImageFile
from sentence_transformers import SentenceTransformer

img_model = SentenceTransformer('clip-ViT-B-16')

# Our text embedding model is aligned to the img_model and maps 50+
# languages to the same vector space
text_model = SentenceTransformer('sentence-transformers/clip-ViT-B-16-multilingual-v1')