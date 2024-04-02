from GraphData import build_data
from DualModel import build_model
#from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import DataLoader

import torch
import torch.optim as optim
import ConfigArgs as args
def main():
    """-----Create DataLoader-----"""
    
    dataset_train = build_data('train')
    dataset_valid = build_data('val')

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_valid)
    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train, num_workers=args.num_workers)
    data_loader_valid = DataLoader(dataset_valid, args.batch_size, sampler=sampler_val, num_workers=args.num_workers)

    for x, y in data_loader_train:
        print(x)
        print(y)
        break

    """==========================="""

    """-----Create Model-----"""

    # dual_model_encoder = build_model()
    # print(dual_model_encoder.eval())
    # """======================"""
    # optimizer = optim.Adam(dual_model_encoder.parameters(), lr=0.0001)

    # # Training loop
    # train_losses = []
    # val_losses = []
    # for epoch in range(args.num_epochs):
    #     dual_model_encoder.train()
    #     total_loss = 0.0
    #     for batch in data_loader_train:
    #         graph_data, text_data = batch
    #         optimizer.zero_grad()
    #         text_embeddings, graph_embeddings = dual_model_encoder(text_data, graph_data)
    #         loss = dual_model_encoder.compute_loss(text_embeddings, graph_embeddings)
    #         loss.backward()
    #         optimizer.step()
    #         total_loss += loss.item()
    #     train_loss = total_loss / len(data_loader_train)
        
    #     # Validation loop
    #     dual_model_encoder.eval()
    #     val_loss = 0.0
    #     with torch.no_grad():
    #         for batch in data_loader_valid:
    #             graph_data, text_data = batch
    #             text_embeddings, graph_embeddings = dual_model_encoder(text_data, graph_data)
    #             loss = dual_model_encoder.compute_loss(text_embeddings, graph_embeddings)
    #             val_loss += loss.item()

    #     val_loss /= len(data_loader_valid)
        
    #     # Log values
    #     train_losses.append(train_loss)
    #     val_losses.append(val_loss)

    #     print(f"Epoch {epoch+1}/{args.num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

if __name__ == "__main__":
    main()