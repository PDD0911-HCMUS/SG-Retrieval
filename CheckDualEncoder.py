import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from PIL import Image
import matplotlib.pyplot as plt

from FashionpediaController.models.backbone import Backbone, Joiner
from FashionpediaController.models.position_encoding import PositionEmbeddingSine
from FashionpediaController.models.transformer import Transformer
from FashionpediaController.models.detr import DETR, build

import os
import pandas as pd
import json
from transformers import BertTokenizer
from torch.nn.utils.rnn import pad_sequence
import ConfigArgs as args
from pathlib import Path

def create_model():
    # position_embedding = PositionEmbeddingSine()
    # backbone = Backbone('resnet50', False, False, False)
    # backbone = Joiner(backbone, position_embedding)
    # backbone.num_channels = 2048
    # transformer = Transformer(d_model=256, dropout=0.1, nhead=8, 
    #                         dim_feedforward=2048,
    #                         num_encoder_layers=6,
    #                         num_decoder_layers=6,
    #                         normalize_before=False,
    #                         return_intermediate_dec=True)

    # model = RelTR(backbone, transformer, llm, num_classes=151, num_rel_classes = 51,
    #             num_entities=100, num_triplets=200)
    model = build()

    return model


transform = T.Compose([
    T.Resize(512),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def sgg_controller(fileName):
    file_name = args.upload + fileName
    path = Path(file_name.replace('.jpg', ''))
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)

    im = Image.open(file_name)
    img = transform(im).unsqueeze(0)
    # propagate through the model
    model = create_model()
    with torch.no_grad():
        # propagate through the model
        out_src = model(img)

    print(out_src.size())

    return out_src



sgg_controller('14.jpg')
# model = create_model()
# print(model.eval())

    