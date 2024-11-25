import os 
from rich import traceback
from datasets.dataV3 import build, custom_collate_fn
from torch.utils.data import DataLoader
from model.Model import build_model
import numpy as np
import torch
from typing import Iterable
from rich.console import Console
from rich.table import Table
from rich.progress import Progress
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def train_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device):
    model.train()
    criterion.train()
    running_loss = 0.0
    with Progress() as progress:
        task = progress.add_task("[cyan]Training...", total=len(data_loader))
        for batch_idx, (que, rev) in enumerate(data_loader):
            que = [{k: v.to(device) for k, v in t.items()} for t in que]
            rev = [{k: v.to(device) for k, v in t.items()} for t in rev]

            output = model(que, rev)
            loss = criterion(output, None)
            weight_dict = criterion.weight_dict
            losses = sum(loss[k] * weight_dict[k] for k in loss.keys() if k in weight_dict)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            running_loss += losses

            progress.update(task, advance=1, description=f"[cyan]Batch {batch_idx+1}/{len(data_loader)} Loss: {losses:.4f}")
            

    epoch_loss = running_loss / len(data_loader)
    return epoch_loss

def validate_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, device: torch.device):
    
    model.eval()
    criterion.eval()
    running_loss = 0.0
    with Progress() as progress:
        task = progress.add_task("[green]Validating...", total=len(data_loader))
        for batch_idx, (que, rev) in enumerate(data_loader):
            que = [{k: v.to(device) for k, v in t.items()} for t in que]
            rev = [{k: v.to(device) for k, v in t.items()} for t in rev]

            output = model(que, rev)
            loss = criterion(output, None)
            weight_dict = criterion.weight_dict
            losses = sum(loss[k] * weight_dict[k] for k in loss.keys() if k in weight_dict)

            running_loss += losses

            progress.update(task, advance=1, description=f"[green]Batch {batch_idx+1}/{len(data_loader)} Loss: {losses:.4f}")

            

    val_loss = running_loss / len(data_loader)
    return val_loss


def main():
    root_data = '/radish/phamd/duypd-proj/SG-Retrieval/Datasets/VisualGenome/'
    root_pth = '/radish/phamd/duypd-proj/SG-Retrieval/ObjectDescriptionV2Controller/ckpt'
    ann_file = root_data + 'Rev.json'
    num_epochs = 100
    lr = 1e-4
    weight_decay = 1e-4
    lr_drop = 50
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)

    print(f"Using device: {device}")

    model, criterion = build_model(d_model=256, dropout=0.1, activation="relu", pretrain = 'bert-base-uncased', device = device)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    
    train_dataset, valid_dataset = build(ann_file)

    dataloader_train = DataLoader(train_dataset, batch_size=32, collate_fn=custom_collate_fn)
    dataloader_valid = DataLoader(valid_dataset, batch_size=32, collate_fn=custom_collate_fn)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_drop)


    console = Console()
    table = Table(title="Training Progress")
    table.add_column("Epoch", justify="center", style="cyan")
    table.add_column("Train Loss", justify="center", style="magenta")
    table.add_column("Validation Loss", justify="center", style="green")

    for epoch in range(num_epochs):
        console.print(f"\n[bold blue]Epoch {epoch + 1}/{num_epochs}[/bold blue]")
        train_loss = train_epoch(model=model,
                                 criterion=criterion, 
                                 data_loader=dataloader_train,
                                 optimizer=optimizer, 
                                 device=device)
        
        val_loss = None
        if dataloader_valid:
            val_loss = validate_epoch(model=model,
                                      criterion=criterion, 
                                      data_loader=dataloader_valid, device=device)
            print(f"Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
        
        table.add_row(str(epoch + 1), f"{train_loss:.4f}", f"{val_loss:.4f}" if val_loss is not None else "N/A")

        if (epoch + 1) % 2 == 0:
            checkpoint_path = f"{root_pth}/model_epoch_{epoch + 1}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Model saved to {checkpoint_path}")

    console.print(table)

if __name__ == "__main__":
    main()