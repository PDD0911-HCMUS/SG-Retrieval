import torch
from torch import nn, Tensor
from transformers import BertModel
import torch.nn.functional as F
from typing import Optional
import copy

class GraphEncoder(nn.Module):
    def __init__(self, hidden_dim, nhead=8, nlayer=6, d_ffn=2048, dropout=0.1, random_erasing_prob=0.3, activation="relu", pre_train = 'bert-base-uncased'):
        super().__init__()

        self.random_erasing_prob = random_erasing_prob
        self.model_embeding = BertModel.from_pretrained(pre_train)
        self.bbox_embed = MLP(4, hidden_dim, hidden_dim, 3)
        self.encoder_layer = TransformerEncoderLayer(hidden_dim, nhead, d_ffn, dropout, activation)

        self.phrase_embed = nn.Sequential(
                            nn.Linear(768, 512),
                            nn.ReLU(),
                            nn.Dropout(dropout),
                            nn.Linear(512, hidden_dim),
                            nn.LayerNorm(hidden_dim))
        
        self.rg_cls = nn.Parameter(torch.zeros(1, hidden_dim)) #[CLS] Token. It will be learned through the training progress
        torch.nn.init.xavier_uniform_(self.rg_cls)
        
        self.layers = _get_clones(self.encoder_layer, nlayer)

        self.nlayer = nlayer
        self.hidden_dim = hidden_dim
        self.nhead = nhead

    def _get_embedding_phrase(self, input_ids, mask):
        word_means = []
        for i,m in zip(input_ids, mask):
            with torch.no_grad():
                outputs = self.model_embeding(i, attention_mask = m)
                word_em = outputs.last_hidden_state
                word_mean = word_em.mean(dim = 1)
                word_means.append(word_mean)
        z = torch.stack([w for w in word_means])
        return z
    
    def _get_embedding_phrase_cls(self, input_ids, mask):
        cls_embeddings = []
        for i, m in zip(input_ids, mask):
            with torch.no_grad():
                outputs = self.model_embeding(i, attention_mask=m)
                # outputs.last_hidden_state có shape [batch, seq_length, hidden_dim]
                # Lấy embedding của token [CLS] (token đầu tiên) cho mỗi sequence
                cls_emb = outputs.last_hidden_state[:, 0, :]
                cls_embeddings.append(cls_emb)
        z = torch.stack(cls_embeddings)
        
        return z

    def forward(self, tgt):

        p_ids = torch.stack([t['phrase_ids'] for t in tgt])
        p_msk = torch.stack([t['phrase_msk'] for t in tgt])

        b_e = torch.stack([self.bbox_embed(t['boxes']) for t in tgt])

        z_t = self.phrase_embed(self._get_embedding_phrase_cls(p_ids, p_msk))

        random_erasing_prob = 0.3

        B, N_phrase, _ = z_t.size()

        padded_mask = (z_t.abs().sum(dim=-1) == 0)

        random_mask  = (torch.rand(B, N_phrase, device=z_t.device) < random_erasing_prob)

        erase_mask = random_mask & (~padded_mask)

        z_t = z_t.masked_fill(erase_mask.unsqueeze(-1), 0)
        b_e = b_e.masked_fill(erase_mask.unsqueeze(-1), 0)

        phrase_pad_mask = (p_msk.sum(dim=-1) == 0) # [B, N_phrase], type bool
        phrase_pad_mask = phrase_pad_mask | erase_mask

        box_mask_list = []
        for t in tgt:
            box_mask = (t['boxes'].sum(dim=-1) == 0)  # [N_box]
            box_mask_list.append(box_mask)
        box_mask = torch.stack(box_mask_list)  # [B, N_box]

        box_mask = box_mask | erase_mask

        cls_token = self.rg_cls.expand(B, -1).unsqueeze(1)

        combined = torch.cat([cls_token, z_t, b_e], dim=1) # [B, 1 + N_phrase + N_box, hidden_dim]

        cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=phrase_pad_mask.device)

        combined_mask = torch.cat([cls_mask, phrase_pad_mask, box_mask], dim=1)  # [B, N_phrase + N_box]

        for layer in self.layers:
            combined = layer(combined, src_key_padding_mask=combined_mask)

        return combined, combined_mask


class TransformerEncoderLayer(nn.Module):

    def __init__(self, hidden_dim, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout, batch_first=True)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(hidden_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, hidden_dim)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
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

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
    
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

def build_graph_encoder(hidden_dim, nhead=8, nlayer=6, d_ffn=2048, dropout=0.1, random_erasing_prob=0.3, activation="relu", pre_train = 'bert-base-uncased'):
    graph_encoder = GraphEncoder(hidden_dim, nhead, nlayer, d_ffn, dropout, random_erasing_prob, activation, pre_train)
    return graph_encoder