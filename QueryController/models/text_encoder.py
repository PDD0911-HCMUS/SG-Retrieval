# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter

from transformers import BertModel, BertConfig, BertTokenizer, BertForMaskedLM
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class MaskedLang(nn.Module):

    def __init__(self,
                 hidden_size=256,
                 pretrained = 'bert-base-uncased'):

        super().__init__()
        
        # Tạo một instance của mô hình BERT với cấu hình đã chỉnh sửa
        self.tokenizer = BertTokenizer.from_pretrained(pretrained)
        self.bert_mlm = BertModel.from_pretrained(pretrained)

        self.reduce = nn.Linear(768, hidden_size)

    def forward(self, feature_maps_sub, feature_maps_obj, text_inputs):
        hs_sub, hs_obj = [], []
        '''dim = layer, batch, q, hidden (6,2,200,256)'''
        for fts, fto in zip(feature_maps_sub, feature_maps_obj): # loop 6 layers
            em_s, em_o = self.compute(text_inputs, fts, fto)
            hs_sub.append(em_s)
            hs_obj.append(em_o)
        
        return torch.stack(hs_sub), torch.stack(hs_obj)
    
    def compute(self, promts, fts, fto):
        ft_sub, ft_obj = [], []
        '''dim feature = batch, q, hidden (2,200,256)'''
        
        for x, y, z in zip(promts, fts, fto): 
            inputs = self.tokenizer(x, return_tensors="pt", padding=True, truncation=True)
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)

            # embeddings = self.bert_mlm.embeddings(input_ids)
            embeddings = self.bert_mlm(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            embeddings = embeddings[:, 0, :]
            embeddings = self.reduce(embeddings)
            mean_em = torch.mean(embeddings, dim=0)

            t1 = y + mean_em
            t2 = z + mean_em
            ft_sub.append(t1)
            ft_obj.append(t2)
        return torch.stack(ft_sub), torch.stack(ft_obj)
    
def build_llm():
    llm = MaskedLang(hidden_size=256, pretrained='bert-base-uncased')
    return llm