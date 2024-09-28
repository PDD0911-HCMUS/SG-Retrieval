# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)

from .cnn_encoder import build_backbone
from .transformer import build_transformer
from transformers import BertTokenizer


class ObjDE(nn.Module):
    def __init__(self, backbone, transformer, seq_length, vocab_size, aux_loss=False):

        super().__init__()
        self.seq_length = seq_length
        self.transformer = transformer
        hidden_dim = transformer.d_model

        self.latent_embed = nn.Embedding(seq_length, hidden_dim)
        self.desc_embed = nn.Linear(hidden_dim, vocab_size)

        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss

    def forward(self, samples: NestedTensor):

        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        src, mask = features[-1].decompose()
        assert mask is not None
        hs = self.transformer(self.input_proj(src), mask, self.latent_embed.weight, pos[-1])[0]
        
        out_desc = self.desc_embed(hs)
        out = {'pred_desc': out_desc[-1]}

        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(out_desc)
        return out


class SetCriterion(nn.Module):
    def __init__(self, vocab_size):
        super(SetCriterion, self).__init__()
        # Sử dụng CrossEntropyLoss với reduction='none' để tính loss cho từng từ
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.vocab_size = vocab_size

    def forward(self, output, target):
        """
        out_desc: Tensor dự đoán từ mô hình, có kích thước (batch_size, seq_length, vocab_size)
        target: Dictionary chứa 'desc_emb' (caption đã được tokenized) và 'desc_msk' (attention mask)
        """
        out_desc = output['pred_desc']
        # Trích xuất 'desc_emb' và 'desc_msk' từ target
        desc_emb = torch.cat([t['desc_emb'] for t in target], dim=0)  # Chuỗi tokenized caption (batch_size, seq_length)
        desc_msk = torch.cat([t['desc_msk'] for t in target], dim=0)  # Attention mask (batch_size, seq_length)

        # Reshape out_desc từ (batch_size, seq_length, vocab_size) thành (batch_size * seq_length, vocab_size)
        batch_size, seq_length, vocab_size = out_desc.size()
        out_desc = out_desc.view(-1, self.vocab_size)  # (batch_size * seq_length, vocab_size)

        # Reshape desc_emb từ (batch_size, seq_length) thành (batch_size * seq_length)
        desc_emb = desc_emb.view(-1)  # (batch_size * seq_length)

        # Tính loss cho từng từ trong chuỗi
        loss = self.criterion(out_desc, desc_emb)  # (batch_size * seq_length)

        # Reshape desc_msk từ (batch_size, seq_length) thành (batch_size * sq_length)
        desc_msk = desc_msk.view(-1)  # (batch_size * seq_length)

        # Chỉ giữ lại loss tại các vị trí có attention_mask = 1 (loại bỏ padding)
        loss = loss * desc_msk  # Vô hiệu hóa loss tại các vị trí padding

        # Tính trung bình loss chỉ trên các từ không bị padding
        return loss.sum() / desc_msk.sum()
    

def build_model(args):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    backbone = build_backbone(args)
    transformer = build_transformer(args)

    model = ObjDE(backbone, transformer, seq_length=args.seq_length, vocab_size=tokenizer.vocab_size)
    criterion = SetCriterion(vocab_size=tokenizer.vocab_size)

    return model, criterion
