import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from .cnn_encoder import build_encoder
from .decoder import build_decoder

class OBDE(nn.Module):
    def __init__(self, cnn_encoder, decodder):
        super(OBDE, self).__init__()
        self.encoder = cnn_encoder
        self.decoder = decodder

    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs

class SetCriterion(nn.Module):
    def __init__(self, pad_idx):
        super(SetCriterion, self).__init__()
        self.pad_idx = pad_idx
        self.criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    def forward(self, outputs, targets):
        # Reshape outputs and targets for loss calculation
        outputs = outputs.view(-1, outputs.size(-1))  # [batch_size * seq_len, vocab_size]
        targets = targets.view(-1)  # [batch_size * seq_len]

        loss = self.criterion(outputs, targets)
        return loss

def build_model(embed_size, vocab_size, num_heads, hidden_dim, num_layers, pad_idx, device):

    cnn_encoder = build_encoder(embed_size)

    decoder = build_decoder(embed_size, vocab_size, num_heads, hidden_dim, num_layers, dropout=0.1)

    model = OBDE(
        cnn_encoder=cnn_encoder,
        decodder=decoder
    )

    criterion = SetCriterion(pad_idx)
    model.to(device=device)
    criterion.to(device=device)

    return model, criterion