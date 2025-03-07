import torch
from torch import nn
from transformers import BertModel

from datasets.dataV3 import build, custom_collate_fn
from torch.utils.data import DataLoader
class GraphEncoder(nn.Module):

    def __init__(self, d_model=512, dropout=0.1, activation="relu", pretrain = 'bert-base-uncased'):
        super().__init__()

        self.model_embeding = BertModel.from_pretrained(pretrain)

        self.activation = _get_activation_fn(activation)

        
        self.sub_embed = nn.Sequential(
                            nn.Linear(768, 512),
                            self.activation,
                            nn.Dropout(dropout),
                            nn.Linear(512, d_model),
                            nn.LayerNorm(d_model))
        
        self.obj_embed = nn.Sequential(
                            nn.Linear(768, 512),
                            self.activation,
                            nn.Dropout(dropout),
                            nn.Linear(512, d_model),
                            nn.LayerNorm(d_model))
        
        self.relation_embed = nn.Sequential(
                            nn.Linear(768, 512),
                            self.activation,
                            nn.Dropout(dropout),
                            nn.Linear(512, d_model),
                            nn.LayerNorm(d_model))
        
        self.trip_embed = nn.Sequential(
                            nn.Linear(768, 512),
                            self.activation,
                            nn.Dropout(dropout),
                            nn.Linear(512, d_model),
                            nn.LayerNorm(d_model))
        
        self.mlp = torch.nn.Sequential(
                    torch.nn.Linear(4*d_model, 512),
                    torch.nn.ReLU(),
                    torch.nn.Linear(512, 256)
        )
        
        self._reset_parameters()
        self.d_model = d_model

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def _get_embedding(self, input_ids, mask):
        word_means = []
        for i,m in zip(input_ids, mask):
            with torch.no_grad():
                outputs = self.model_embeding(i, attention_mask = m)
                word_em = outputs.last_hidden_state.squeeze(0)
                word_means.append(word_em)
        z = torch.stack([w for w in word_means])
        return z
    
    def _get_embedding_trip(self, input_ids, mask):
        word_means = []
        for i,m in zip(input_ids, mask):
            with torch.no_grad():
                outputs = self.model_embeding(i, attention_mask = m)
                word_em = outputs.last_hidden_state
                word_mean = word_em.mean(dim = 1)
                word_means.append(word_mean)
        z = torch.stack([w for w in word_means])
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

        z_s = self.sub_embed(self._get_embedding(s, s_msk))
        z_o = self.obj_embed(self._get_embedding(o, o_msk))
        z_r = self.relation_embed(self._get_embedding(r, r_msk))

        z_t = self.trip_embed(self._get_embedding_trip(t, t_msk))

        combined = torch.cat([z_s, z_o, z_r, z_t], dim=2)
        output = self.mlp(combined)

        return output

def _get_activation_fn(activation):
    if activation == "relu":
        return nn.ReLU()
    if activation == "gelu":
        return nn.GELU()
    if activation == "glu":
        return nn.GLU()
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
