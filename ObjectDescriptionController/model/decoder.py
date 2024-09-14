import math
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, embed_size)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2) * -(math.log(10000.0) / embed_size))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        return x + self.encoding[:, :x.size(1), :].to(x.device)


class TransformerDecoder(nn.Module):
    def __init__(self, embed_size, vocab_size, num_heads, hidden_dim, num_layers, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.positional_encoding = PositionalEncoding(embed_size)
        self.transformer_decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_size, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout
        )
        self.transformer_decoder = nn.TransformerDecoder(self.transformer_decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, features, captions):
        captions_embed = self.embedding(captions)  # [batch_size, seq_len, embed_size]
        captions_embed = self.positional_encoding(captions_embed)  # [batch_size, seq_len, embed_size]
        captions_embed = self.dropout(captions_embed)
        
        # Transformer Decoder
        captions_output = self.transformer_decoder(captions_embed.permute(1, 0, 2), features.unsqueeze(0))
        output = self.fc_out(captions_output.permute(1, 0, 2))
        return output

def build_decoder(embed_size, 
                  vocab_size, 
                  num_heads, 
                  hidden_dim, 
                  num_layers, 
                  dropout=0.1):
    
    decoder = TransformerDecoder(
        embed_size, 
        vocab_size, 
        num_heads, 
        hidden_dim, 
        num_layers, 
        dropout)
    
    return decoder