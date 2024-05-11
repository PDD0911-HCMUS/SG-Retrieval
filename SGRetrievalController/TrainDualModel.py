# from GraphData import build_data, collate_fn
from VGData import build_data, collate_fn
from RelTRnBert import build_model
from torch.utils.data import DataLoader
import numpy as np
import random
import torch
import torch.optim as optim
import ConfigArgs as args
from pathlib import Path
from tqdm import tqdm
import json


def main():
    # fix the seed for reproducibility
    output_dir = 'ckpt/'
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    """-----Create DataLoader-----"""
    
    dataset_train = build_data('train')
    dataset_valid = build_data('val')

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_valid)
    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train, num_workers=args.num_workers, collate_fn=collate_fn)
    data_loader_valid = DataLoader(dataset_valid, args.batch_size, sampler=sampler_val, num_workers=args.num_workers, collate_fn=collate_fn)

    """==========================="""

    """-----Create Model-----"""

    dual_model_encoder, dot_loss = build_model(
        num_projection_layers=1,
        projection_dims=256,
        dropout_rate=0.1
    )
    # print(dual_model_encoder.eval())
    device = torch.device(torch.device(args.device if torch.cuda.is_available() else "cpu"))
    
    dual_model_encoder.to(device=device)
    dot_loss.to(device=device)
    optimizer = optim.Adam(dual_model_encoder.parameters(), lr=0.0001)

    """======================"""
    

    # Training loop
    train_losses = []
    val_losses = []
    for epoch in range(args.num_epochs):
        print(f"Epoch {epoch+1}/{args.num_epochs}")
        dual_model_encoder.train()
        dot_loss.train()
        total_loss = 0.0
        for graph_data, text_data in tqdm(data_loader_train):
            graph_data = graph_data.to(device)
            text_data = text_data.to(device)

            optimizer.zero_grad()
            graph_embeddings, text_embeddings = dual_model_encoder(graph_data, text_data)
            loss = dot_loss(text_embeddings, graph_embeddings)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_loss = total_loss / len(data_loader_train)
        
        # Validation loop
        dual_model_encoder.eval()
        dot_loss.eval()
        val_loss = 0.0
        with torch.no_grad():
            for graph_data, text_data in tqdm(data_loader_valid):
                graph_data = graph_data.to(device)
                text_data = text_data.to(device)
                
                graph_embeddings, text_embeddings = dual_model_encoder(graph_data, text_data)
                loss = dot_loss(text_embeddings, graph_embeddings)
                val_loss += loss.item()
        
        dual_model_encoder.reltr.state_dict()
        dual_model_encoder.bert.state_dict()

        val_loss /= len(data_loader_valid)
        
        # Log values
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        log_stats = {**{f'train_loss': train_loss},
                     **{f'val_loss': val_loss},
                     'epoch': epoch
                     }
        print(f"Epoch {epoch+1}/{args.num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        with (Path(output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

        torch.save(dual_model_encoder.reltr.state_dict(), output_dir + 'reltr_weights_' + str(epoch) + '.pth')
        torch.save(dual_model_encoder.bert.state_dict(), output_dir + 'bert_weights_' + str(epoch) + '.pth')

if __name__ == "__main__":
    main()