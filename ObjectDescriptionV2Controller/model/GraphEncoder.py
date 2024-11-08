import copy
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor

class GraphEncoder(nn.Module):
    
    def __init__(self, class_to_idx, rel_to_idx, node_bbox=False, d_model=512, nhead=8, nlayer=6, d_ffn=2048, dropout=0.1, activation="relu"):
        super().__init__()
        self.class_to_idx = class_to_idx
        self.rel_to_idx = rel_to_idx
        self.node_bbox = node_bbox
        self.d_model = d_model
        self.nhead = nhead
        
        # Embeddings cho node và edge
        self.entity_embed = nn.Embedding(len(class_to_idx), d_model)
        self.relation_embed = nn.Embedding(len(rel_to_idx), d_model)
        
        # Thêm embedding cho bounding box nếu cần
        if self.node_bbox:
            self.box_embed = nn.Sequential(
                nn.Linear(4, 128),
                nn.BatchNorm1d(128),
                nn.Linear(128, d_model)
            )
            self.norm = nn.LayerNorm(d_model)
        
        # Token đặc trưng cho scene graph
        self.graph_cls = nn.Parameter(torch.zeros(1, d_model))
        
        # Thêm các embedding vị trí cho node và edge
        self.node_encodings = nn.Parameter(torch.zeros(10, d_model))
        
        # Xây dựng Transformer Encoder
        encoder_layer = TransformerEncoderLayer(d_model, nhead, d_ffn, dropout, activation)
        self.layers = nn.ModuleList([encoder_layer for _ in range(nlayer)])
        
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, graphs):
        nodes = []
        edges = []
        
        for g in graphs:
            # Mã hóa từng triplet trong trường `trip`
            triplets = g["trip"]
            node_embeddings = []
            edge_embeddings = []

            for triplet in triplets:
                # Tách các thành phần của triplet
                obj1, rel, obj2 = triplet.split(' ')[0], triplet.split(' ')[1], triplet.split(' ')[2]
                
                # Lấy index của các đối tượng và mối quan hệ
                obj1_idx = self.class_to_idx.get(obj1, 0)  # Nếu không tìm thấy, gán giá trị 0 (PAD)
                rel_idx = self.rel_to_idx.get(rel, 0)      # Nếu không tìm thấy, gán giá trị 0
                obj2_idx = self.class_to_idx.get(obj2, 0)  # Nếu không tìm thấy, gán giá trị 0
                
                # Mã hóa đối tượng và mối quan hệ
                node_embeddings.append(self.entity_embed(torch.tensor(obj1_idx)))
                node_embeddings.append(self.entity_embed(torch.tensor(obj2_idx)))
                edge_embeddings.append(self.relation_embed(torch.tensor(rel_idx)))
            
            # Chuyển danh sách các embedding thành tensor và thêm padding nếu cần
            node_embedding = torch.stack(node_embeddings)
            edge_embedding = torch.stack(edge_embeddings)
            
            # Padding cho nodes (10 node) và edges (7 edges)
            if node_embedding.size(0) < 10:
                pad_length = 10 - node_embedding.size(0)
                node_embedding = F.pad(node_embedding, (0, 0, 0, pad_length))
            
            if edge_embedding.size(0) < 7:
                pad_length = 7 - edge_embedding.size(0)
                edge_embedding = F.pad(edge_embedding, (0, 0, 0, pad_length))
                
            nodes.append(node_embedding)
            edges.append(edge_embedding)
        
        # Chuyển đổi các danh sách thành tensor
        nodes = torch.stack(nodes)
        edges = torch.stack(edges)
        
        # Tạo mã hóa vị trí cho node và edge
        nodes_encodings = self.node_encodings.unsqueeze(0).repeat(len(graphs), 1, 1)
        edges_encodings = self.node_encodings[:edges.size(1)].unsqueeze(0).repeat(len(graphs), 1, 1)
        
        # Mask cho node và edge
        nodes_mask = torch.zeros(nodes.size(0), nodes.size(1), dtype=torch.bool)
        edges_mask = torch.zeros(edges.size(0), edges.size(1), dtype=torch.bool)
        
        # Token đặc trưng cho scene graph
        graph_cls = self.graph_cls.unsqueeze(0).repeat(len(graphs), 1, 1)
        graph_mask = torch.zeros([len(graphs), 1], dtype=torch.bool)
        
        # Chuẩn bị đầu vào cho Transformer Encoder
        output = torch.cat([graph_cls, nodes, edges], dim=1)
        pos = torch.cat([torch.zeros([len(graphs), 1, self.d_model]), nodes_encodings, edges_encodings], dim=1)
        mask = torch.cat([graph_mask, nodes_mask, edges_mask], dim=1)
        
        # Truyền qua các lớp Transformer Encoder
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
    
def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")