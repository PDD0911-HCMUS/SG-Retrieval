import torch
from torch import nn, Tensor
import torch.nn.functional as F
from transformers import BertModel
from typing import Optional
import copy

def _get_activation_fn(activation: str):
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}")

def _get_clones(module: nn.Module, N: int):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class TransformerEncoderLayer(nn.Module):
    def __init__(self, hidden_dim, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(hidden_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)

    def forward(self, src: Tensor, src_key_padding_mask: Optional[Tensor] = None, pos: Optional[Tensor] = None):
        q = k = src if pos is None else src + pos
        src2 = self.self_attn(q, k, value=src, key_padding_mask=src_key_padding_mask)[0]
        src = self.norm1(src + self.dropout1(src2))
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = self.norm2(src + self.dropout2(src2))
        return src

class GraphEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        nhead: int = 8,
        nlayer: int = 6,
        d_ffn: int = 2048,
        dropout: float = 0.1,
        random_delete_prob: float = 0.3,
        freeze_bert: bool = True,
        activation: str = "relu",
        pretrained_model: str = 'bert-base-uncased',
    ):
        super().__init__()
        self.random_delete_prob = random_delete_prob

        # 1) BERT for phrase embedding
        self.bert = BertModel.from_pretrained(pretrained_model)
        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False

        # 2) Project BERT hidden_size → hidden_dim
        bert_dim = self.bert.config.hidden_size
        self.phrase_embed = nn.Sequential(
            nn.Linear(bert_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # 3) Two [CLS] tokens: one for original, one for erased
        self.sg_cls = nn.Parameter(torch.zeros(1, hidden_dim))
        self.sg_cls_erased = nn.Parameter(torch.zeros(1, hidden_dim))
        nn.init.xavier_uniform_(self.sg_cls)
        nn.init.xavier_uniform_(self.sg_cls_erased)

        # 4) Transformer encoder stack
        encoder_layer = TransformerEncoderLayer(hidden_dim, nhead, d_ffn, dropout, activation)
        self.layers = _get_clones(encoder_layer, nlayer)

    def _get_embedding_phrase_cls(self, input_ids: Tensor, attention_mask: Tensor):
        """
        input_ids, attention_mask: [B, N, L]
        returns: [B, N, bert_hidden]
        """
        B, N, L = input_ids.shape
        flat_ids = input_ids.view(B * N, L)
        flat_mask = attention_mask.view(B * N, L)

        valid = flat_mask.sum(dim=1) > 0
        valid_ids = flat_ids[valid]
        valid_mask = flat_mask[valid]

        cls_emb = torch.zeros(B * N, self.bert.config.hidden_size, device=input_ids.device)
        if valid.any():
            out = self.bert(valid_ids, attention_mask=valid_mask, return_dict=True)
            cls_emb_valid = out.last_hidden_state[:, 0, :]
            cls_emb[valid] = cls_emb_valid

        return cls_emb.view(B, N, -1)

    def forward(self, batch):
        """
        batch: list of dicts, each with
          'trip_ids': Tensor[N, L],
          'trip_mask': Tensor[N, L]
        returns:
          zt_e:    [B, 1+N, hidden_dim] original
          zt_r_e:  [B, 1+N, hidden_dim] with random-deletion
          mask_e:  [B, 1+N] key_padding_mask for original
          mask_r:  [B, 1+N] key_padding_mask for erased
        """
        # 1) stack inputs
        t_ids = torch.stack([x['trip_ids'] for x in batch])   # [B, N, L]
        t_msk = torch.stack([x['trip_mask'] for x in batch])  # [B, N, L]

        # 2) get BERT [CLS] for each phrase
        cls_emb = self._get_embedding_phrase_cls(t_ids, t_msk)  # [B, N, bert_hidden]
        z_t = self.phrase_embed(cls_emb)                       # [B, N, hidden_dim]

        B, N, D = z_t.shape

        # 3) compute padded mask
        pad_mask = (t_msk.sum(dim=-1) == 0)  # [B, N]

        # 4) random delete only in training
        if self.training and self.random_delete_prob > 0:
            rand = torch.rand(B, N, device=z_t.device)
            delete_mask = (rand < self.random_delete_prob) & (~pad_mask)
        else:
            delete_mask = torch.zeros_like(pad_mask)

        # 5) build erased version
        z_t_erased = z_t.masked_fill(delete_mask.unsqueeze(-1), 0)

        # 6) prepend CLS tokens
        cls       = self.sg_cls.expand(B, -1).unsqueeze(1)           # [B,1,D]
        cls_er    = self.sg_cls_erased.expand(B, -1).unsqueeze(1)    # [B,1,D]
        zt_e      = torch.cat([cls,       z_t], dim=1)              # [B,1+N,D]
        zt_r_e    = torch.cat([cls_er,    z_t_erased], dim=1)       # [B,1+N,D]

        # 7) build key_padding masks
        cls_mask  = torch.zeros(B, 1, dtype=torch.bool, device=pad_mask.device)
        mask_e    = torch.cat([cls_mask, pad_mask],   dim=1)        # [B,1+N]
        mask_r    = torch.cat([cls_mask, pad_mask|delete_mask], dim=1)

        # 8) pass through Transformer layers
        for layer in self.layers:
            zt_e   = layer(zt_e,   src_key_padding_mask=mask_e)
            zt_r_e = layer(zt_r_e, src_key_padding_mask=mask_r)

        # return zt_e, zt_r_e, mask_e, mask_r
        return zt_e, zt_r_e, mask_e

def build_graph_encoder(
    hidden_dim: int,
    nhead: int = 8,
    nlayer: int = 6,
    d_ffn: int = 2048,
    dropout: float = 0.1,
    random_delete_prob: float = 0.3,
    freeze_bert: bool = True,
    activation: str = "relu",
    pretrained_model: str = 'bert-base-uncased',
):
    return GraphEncoder(
        hidden_dim, nhead, nlayer, d_ffn, dropout,
        random_delete_prob, freeze_bert, activation, pretrained_model
    )
