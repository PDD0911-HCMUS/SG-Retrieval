import torch
import torch.nn as nn

class RegionDecoder(nn.Module):
    def __init__(self, hidden_dim, tokenizer, max_len, nhead, nlayer, dropout):
        super().__init__() 
        
        self.hidden_dim = hidden_dim
        self.vocab_size = tokenizer.vocab_size
        self.max_len = max_len
        self.nhead = nhead
        self.nlayer = nlayer
        self.dropout = dropout

        # Token embedding
        self.token_embed = nn.Embedding(self.vocab_size, self.hidden_dim)

        # Positional encoding
        self.pos_embed = nn.Embedding(self.max_len, self.hidden_dim)

        # Project region_feat to memory format
        self.memory_proj = nn.Linear(self.hidden_dim, self.hidden_dim)

        decoder_layer = nn.TransformerDecoderLayer(d_model=self.hidden_dim, nhead=self.nhead, dropout=self.dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=self.nlayer)

        self.vocab_proj = nn.Linear(self.hidden_dim, self.vocab_size)

    def forward(self, region_feat, tgt_tokens):
        """
        region_feat: [B, Nq, D]
        tgt_tokens:  [B, N, seq_len]
        return:      [B, Nq, seq_len, vocab_size]
        """
        B, Nq, D = region_feat.shape
        seq_len = tgt_tokens.shape[-1]

        # (1) Flatten batch & query dims
        region_feat = region_feat.view(B * Nq, D) # [B, Nq, D] -> [B*Nq, D]
        tgt_tokens = tgt_tokens.view(B * Nq, seq_len) # [B, N, seq_len] -> [B*N, seq_len]

        # (2) Embedding
        tgt_emb = self.token_embed(tgt_tokens) # [B*Nq, seq_len, D]

        # (3) Add positional embedding
        pos_ids = torch.arange(seq_len, device=tgt_tokens.device).unsqueeze(0)
        pos_emb = self.pos_embed(pos_ids)                           
        tgt_emb = tgt_emb + pos_emb

        # (4) Prepare memory
        memory = self.memory_proj(region_feat).unsqueeze(0) # [1, B*Nq, D]

        # (5) Transformer Decoder
        tgt_emb = tgt_emb.permute(1, 0, 2) # [seq_len, B*Nq, D]
        dec_out = self.decoder(tgt_emb, memory) # [seq_len, B*Nq, D]

        # (6) Output
        logits = self.vocab_proj(dec_out.permute(1, 0, 2)) # [B*Nq, seq_len, vocab]
        logits = logits.view(B, Nq, seq_len, self.vocab_size) # [B, Nq, seq_len, vocab_size]
        return logits