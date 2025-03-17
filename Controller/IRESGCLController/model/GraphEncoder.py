import copy
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor

class GraphEncoder(nn.Module):

    def __init__(self, d_model=512, nhead=8, nlayer=6, d_ffn=2048,
                 dropout=0.1, activation="relu"):
        super().__init__()
        self.graph_cls = nn.Parameter(torch.zeros(1, d_model))

        self.entity_embed = nn.Embedding(152, d_model)
        self.sub_embed = nn.Embedding(152, d_model)
        self.obj_embed = nn.Embedding(152, d_model)
        self.relation_embed = nn.Embedding(52, d_model)

        self.node_encodings = nn.Parameter(torch.zeros(10, d_model))
        self.node_encodings_s = nn.Parameter(torch.zeros(10, d_model))
        self.node_encodings_o = nn.Parameter(torch.zeros(10, d_model))
        self.edge_encodings = nn.Parameter(torch.zeros(10, d_model))

        encoder_layer = TransformerEncoderLayer(d_model, nhead, d_ffn, dropout, activation)
        self.layers = _get_clones(encoder_layer, nlayer)
        self.nlayer = nlayer

        self._reset_parameters()
        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, graphs):

        nodes = torch.stack([self.entity_embed(g["labels"]) for g in graphs])
        nodes_s = torch.stack([self.entity_embed(g["sub_labels"]) for g in graphs])
        nodes_o = torch.stack([self.entity_embed(g["obj_labels"]) for g in graphs])
        edges = torch.stack([self.relation_embed(g["rel_labels"]) for g in graphs])

        nodes_mask = torch.stack([g["labels"] == 151 for g in graphs])
        nodes_mask_s = torch.stack([g["sub_labels"] == 151 for g in graphs])
        nodes_mask_o = torch.stack([g["obj_labels"] == 151 for g in graphs])
        edges_mask = torch.stack([g["rel_labels"] == 51 for g in graphs])

        nodes_encodings = self.node_encodings.unsqueeze(0).repeat(len(graphs), 1, 1)
        nodes_encodings_s = self.node_encodings.unsqueeze(0).repeat(len(graphs), 1, 1)
        nodes_encodings_o = self.node_encodings.unsqueeze(0).repeat(len(graphs), 1, 1)
        edges_encodings = self.edge_encodings.unsqueeze(0).repeat(len(graphs), 1, 1)
        

        graph_cls = self.graph_cls.unsqueeze(0).repeat(len(graphs), 1, 1)
        graph_mask = torch.zeros([len(graphs), 1], dtype=torch.bool, device=graphs[0]["labels"].device)

        output = torch.cat(
            [
                graph_cls, 
                nodes, 
                nodes_s,
                nodes_o,
                edges
            ], dim=1)
        pos = torch.cat(
            [
                torch.zeros([len(graphs), 1, self.d_model], device=graphs[0]["labels"].device), 
                nodes_encodings, 
                nodes_encodings_s,
                nodes_encodings_o,
                edges_encodings
            ], dim=1)
        mask = torch.cat(
            [
                graph_mask, 
                nodes_mask,
                nodes_mask_s,
                nodes_mask_o,
                edges_mask
            ], dim=1)

        for layer in self.layers:
            output = layer(output, src_key_padding_mask=mask, pos=pos)

        return output, mask, pos

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
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
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
