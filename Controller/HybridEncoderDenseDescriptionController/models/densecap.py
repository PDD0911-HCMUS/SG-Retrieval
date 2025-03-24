import torch
import torch.nn as nn
from transformers import BertModel
from densecap_position_encoding import PositionalEncoding

class BERTEmbeddingWithPositional(nn.Module):
    def __init__(self, from_pretrained, seq_length):
        super().__init__()
        
        # Khởi tạo BERT model và Positional Encoding
        self.bert = BertModel.from_pretrained(from_pretrained)
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        with torch.no_grad():
            outputs = self.bert(input_ids, attention_mask=attention_mask)
            word_embeddings = outputs.last_hidden_state
        
        return word_embeddings
    

if __name__ == '__main__':
    pe = PositionalEncoding(d_model=768, max_len=7)
    be = BERTEmbeddingWithPositional(
        from_pretrained='bert-base-uncased',
        seq_length=7
    )

    desc_ids = torch.tensor([[ 101, 2023, 2482, 2003, 2317, 1012, 2023, 7381, 2038, 2665, 4303, 1012,
         5725, 2422, 1997, 2482, 1012, 2023, 2482, 2003, 2317, 1012, 5725, 2422,
         1997, 2482, 1012,  102,    0,    0,    0,    0,    0,    0,    0,    0,
            0,    0,    0,    0]])
    
    desc_msk = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    # desc_ids = desc_ids.unsqueeze(0)
    # desc_msk = desc_msk.unsqueeze(0)
    out = be(desc_ids, desc_msk)
    print(out.size())