import copy
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from transformers import BertModel
class GraphEncoder(nn.Module):

    def __init__(self, d_model=512, nhead=8, nlayer=6, d_ffn=2048,
                 dropout=0.1, activation="relu", pretrain = 'bert-base-uncased'):
        super().__init__()

        self.model_embeding = BertModel.from_pretrained(pretrain)

        self.activation = _get_activation_fn(activation)

        
        self.sub_embed = nn.Sequential(
                            nn.Linear(768, 512),
                            self.activation,
                            nn.Dropout(0.3),
                            nn.Linear(512, d_model),
                            nn.LayerNorm(d_model))
        
        self.obj_embed = nn.Sequential(
                            nn.Linear(768, 512),
                            self.activation,
                            nn.Dropout(0.3),
                            nn.Linear(512, d_model),
                            nn.LayerNorm(d_model))
        
        self.relation_embed = nn.Sequential(
                            nn.Linear(768, 512),
                            self.activation,
                            nn.Dropout(0.3),
                            nn.Linear(512, d_model),
                            nn.LayerNorm(d_model))
        
        self.trip_embed = nn.Sequential(
                            nn.Linear(768, 512),
                            self.activation,
                            nn.Dropout(0.3),
                            nn.Linear(512, d_model),
                            nn.LayerNorm(d_model))
        
        self._reset_parameters()
        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def _get_word_embedding(self, input_ids, mask):
        word_means = []
        for i,m in zip(input_ids, mask):
            with torch.no_grad():
                outputs = self.model_embeding(i, attention_mask = m)
                word_em = outputs.last_hidden_state
                word_mean = word_em.mean(dim = 1)
                word_means.append(word_mean)
        z = torch.stack([w for w in word_means])
        # print(z.size())
        return z

    def forward(self, graphs):

        s = torch.stack([g["sub_labels"] for g in graphs])
        s_msk = torch.stack([g["sub_labels_msk"] for g in graphs])
        o = torch.stack([g["obj_labels"] for g in graphs])
        o_msk = torch.stack([g["obj_labels_msk"] for g in graphs])
        r = torch.stack([g["rel_labels"] for g in graphs])
        r_msk = torch.stack([g["rel_labels_msk"] for g in graphs])

        t = torch.stack([g["trip"] for g in graphs])
        t_msk = torch.stack([g["trip_msk"] for g in graphs])

        z_s = self.sub_embed(self._get_word_embedding(s, s_msk))
        z_o = self.sub_embed(self._get_word_embedding(o, o_msk))
        z_r = self.sub_embed(self._get_word_embedding(r, r_msk))

        z_t = self.trip_embed(self._get_word_embedding(t, t_msk))



        return

class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, src, src_key_padding_mask: Optional[Tensor] = None, pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src
    
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    if activation == "relu":
        return nn.ReLU()
    if activation == "gelu":
        return nn.GELU()
    if activation == "glu":
        return nn.GLU()
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

# if __name__ == "__main__":
#     sample_data = {
#         'sub_labels': torch.tensor([15, 151, 151, 151, 151, 151, 151, 151, 151, 151]),
#         'obj_labels': torch.tensor([126, 151, 151, 151, 151, 151, 151, 151, 151, 151]),
#         'rel_labels': torch.tensor([31, 51, 51, 51, 51, 51, 51, 51, 51, 51]),
#         'labels': torch.tensor([15, 126, 151, 151, 151, 151, 151, 151, 151, 151])
#     }

#     graphs = [sample_data]
#     graph_encoder = GraphEncoder(d_model=512, nhead=8, nlayer=6, d_ffn=2048, dropout=0.1, activation="relu")
#     output, mask, pos = graph_encoder(graphs)
