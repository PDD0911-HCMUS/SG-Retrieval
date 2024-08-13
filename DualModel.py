import torch
import torch.nn as nn
import torch.nn.functional as F
import ConfigArgs as args
# from torch_geometric.nn import GCNConv, global_mean_pool
# from transformers import BertModel

def project_embeddings(embeddings, num_projection_layers, projection_dims, dropout_rate):
    projection_layer = nn.Linear(embeddings.size(1), projection_dims).to(args.device)
    norm_layer = nn.LayerNorm(projection_dims).to(args.device)
    projected_embeddings = projection_layer(embeddings)
    for _ in range(num_projection_layers):
        x = F.relu(projected_embeddings)
        x = nn.Linear(projection_dims, projection_dims).to(args.device)(x)
        x = F.dropout(x, p=dropout_rate)
        x = projected_embeddings + x
        projected_embeddings = norm_layer(x)
    return projected_embeddings

class GraphEncoder(nn.Module):
    def __init__(self, num_projection_layers, projection_dims, dropout_rate, pretrained=True):
        super(GraphEncoder, self).__init__()
        self.bert = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-uncased').to(args.device)
        self.tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-uncased')
        self.num_projection_layers = num_projection_layers
        self.projection_dims = projection_dims
        self.dropout_rate = dropout_rate

    def forward(self, x):
        # Tiền xử lý dữ liệu văn bản sử dụng tokenizer của BERT
        #inputs = self.tokenizer(x, return_tensors='pt', padding=True, truncation=True)
        outputs = self.bert(x)
        pooled_output = outputs['pooler_output']
        pooled_output = pooled_output.to(args.device)
        x = project_embeddings(pooled_output, self.num_projection_layers, self.projection_dims, self.dropout_rate)
        return x
    
class TextEncoder(nn.Module):
    def __init__(self, num_projection_layers, projection_dims, dropout_rate, pretrained=True):
        super(TextEncoder, self).__init__()
        self.bert = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-uncased').to(args.device)
        self.tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-uncased')
        self.num_projection_layers = num_projection_layers
        self.projection_dims = projection_dims
        self.dropout_rate = dropout_rate

    def forward(self, x):
        # Tiền xử lý dữ liệu văn bản sử dụng tokenizer của BERT
        #inputs = self.tokenizer(x, return_tensors='pt', padding=True, truncation=True)
        outputs = self.bert(x)
        pooled_output = outputs['pooler_output']
        pooled_output = pooled_output.to(args.device)
        x = project_embeddings(pooled_output, self.num_projection_layers, self.projection_dims, self.dropout_rate)
        return x


class DualEncoder(nn.Module):
    def __init__(self, text_encoder, graph_encoder, temperature=1.0):
        super(DualEncoder, self).__init__()
        self.text_encoder = text_encoder
        self.graph_encoder = graph_encoder
        self.temperature = temperature
        self.loss = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, text_data, graph_data):
        caption_embeddings = self.text_encoder(text_data)
        graph_embeddings = self.graph_encoder(graph_data)
        return caption_embeddings, graph_embeddings

    def compute_loss(self, caption_embeddings, graph_embeddings):
        logits = torch.matmul(caption_embeddings, graph_embeddings.t()) / self.temperature
        graph_similarity = torch.matmul(graph_embeddings, graph_embeddings.t())
        captions_similarity = torch.matmul(caption_embeddings, caption_embeddings.t())
        targets = (captions_similarity + graph_similarity) / (2 * self.temperature)
        captions_loss = self.loss(logits, targets)
        graph_loss = self.loss(logits.t(), targets.t())
        return (captions_loss + graph_loss) / 2


def build_model():
    text_encoder = TextEncoder(
        num_projection_layers=1, 
        projection_dims=256, 
        dropout_rate=0.1)
    
    graph_encoder = GraphEncoder(
        # num_features=300,
        num_projection_layers=1, 
        projection_dims=256, 
        dropout_rate=0.1)
    
    dual_model_encoder = DualEncoder(
        text_encoder=text_encoder,
        graph_encoder=graph_encoder,
        temperature=0.05
    )
    return dual_model_encoder