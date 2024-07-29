import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from PIL import Image
import matplotlib.pyplot as plt

from SGGController.RelTR.models.backbone import Backbone, Joiner
from SGGController.RelTR.models.position_encoding import PositionEmbeddingSine
from SGGController.RelTR.models.transformer import Transformer
from SGGController.RelTR.models.reltr import RelTR
from SGGController.RelTR.models.llm import MaskedLang

import os
import pandas as pd
import json
from transformers import BertTokenizer
from torch.nn.utils.rnn import pad_sequence
import ConfigArgs as args
from pathlib import Path

def create_model():
    position_embedding = PositionEmbeddingSine(128, normalize=True)
    backbone = Backbone('resnet50', False, False, False)
    backbone = Joiner(backbone, position_embedding)
    backbone.num_channels = 2048
    llm = MaskedLang(hidden_size=256, pretrained='bert-base-uncased')
    transformer = Transformer(d_model=256, dropout=0.1, nhead=8, 
                            dim_feedforward=2048,
                            num_encoder_layers=6,
                            num_decoder_layers=6,
                            normalize_before=False,
                            return_intermediate_dec=True)

    model = RelTR(backbone, transformer, llm, num_classes=151, num_rel_classes = 51,
                num_entities=100, num_triplets=200)

    # The checkpoint is pretrained on Visual Genome
    ckpt = torch.load('ckpt/checkpoint0195.pth', map_location='cpu')
    model.load_state_dict(ckpt['model'])
    model.eval()

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
        out, out_src = model(img, None)

    print(out_src.size())

    return out_src



sgg_controller('12.jpg')

    