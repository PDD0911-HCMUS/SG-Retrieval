from Dataset.create_data import build_data
from .create_model import build_model
import torch
from torch.utils.data import DataLoader
import datetime
import time
import torch.optim as optim
from rich.console import Console
from rich.table import Table
from rich.progress import track
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2" 

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    console = Console()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.log(f"[bold green]Using device:[/bold green] {device}")

    # Dataset Configuration
    batch_size = 4
    ratio = 0.8
    csv_file = '/radish/phamd/duypd-proj/SG-Retrieval/khangTA/Dataset/final_seg_words.csv'

    # Model Configuration
    pretrained = 'vinai/phobert-base-v2'
    num_classes = 8
    drop_out = 0.1
    start_epoch = 0
    epochs = 50
    learning_rate = 1e-4
    step_size = 10
    gamma = 0.1   

    console.log("[bold yellow]Preparing Dataset[/bold yellow]")
    train_dataset, valid_dataset = build_data(csv_file, ratio)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    console.log("[bold cyan]Data Summary:[/bold cyan]")
    console.log(f"Total samples in train set: {len(train_dataset)}")
    console.log(f"Total samples in validation set: {len(valid_dataset)}")
    console.log(f"Batch size: {batch_size}")
    console.log(f"Total train batches: {len(train_loader)}")
    console.log(f"Total validation batches: {len(valid_loader)}")

    console.log("[bold yellow]Preparing Model[/bold yellow]")
    model, criterion = build_model(from_pretrained=pretrained, num_classes=num_classes, drop_out=drop_out)
    model.to(device)
    criterion.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    console.log(f"\n[bold green]Total trainable parameters:[/bold green] {count_parameters(model)}")

    train_losses = []
    val_losses = []
    val_accuracies = []

    console.log("[bold yellow]Start training[/bold yellow]")
    start_time = time.time()

    for epoch in range(start_epoch, epochs):
        console.log(f"[bold blue]Training Epoch [{epoch+1}/{epochs}][/bold blue]")
        model.train()
        criterion.train()
        total_train_loss = 0

        for samples, msks, labels in track(train_loader, description="Training"):
            samples = samples.to(device)
            msks = msks.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(samples, msks)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        total_val_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            console.log(f"[bold magenta]Validation Epoch [{epoch+1}/{epochs}][/bold magenta]")
            for samples, msks, labels in track(valid_loader, description="Validating"):
                samples = samples.to(device)
                msks = msks.to(device)
                labels = labels.to(device)

                outputs = model(samples, msks)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()

                _, predicted_classes = torch.max(outputs, 1)
                correct_predictions += (predicted_classes == labels).sum().item()
                total_predictions += labels.size(0)

        avg_val_loss = total_val_loss / len(valid_loader)
        val_accuracy = correct_predictions / total_predictions

        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)

        lr_scheduler.step()

        if (epoch + 1) % 2 == 0:
            model_path = f"/radish/phamd/duypd-proj/SG-Retrieval/khangTA/ClassificationController/ckpt_v2_8classes/model_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), model_path)
            console.log(f"[bold green]Model saved at:[/bold green] {model_path}")

        console.log(f"Epoch [{epoch+1}/{epochs}], "
                    f"Train Loss: {avg_train_loss:.4f}, "
                    f"Validation Loss: {avg_val_loss:.4f}, "
                    f"Validation Accuracy: {val_accuracy:.4f}, "
                    f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    console.log(f"[bold green]Training time {total_time_str}[/bold green]")

    # Summary Table
    table = Table(title="Training Summary")
    table.add_column("Epoch", justify="center")
    table.add_column("Train Loss", justify="right")
    table.add_column("Validation Loss", justify="right")
    table.add_column("Validation Accuracy", justify="right")

    for epoch in range(epochs):
        table.add_row(str(epoch+1), 
                      f"{train_losses[epoch]:.4f}", 
                      f"{val_losses[epoch]:.4f}", 
                      f"{val_accuracies[epoch]:.4f}")
    
    console.print(table)

    # Save log to a file
    with open("training_log.txt", "w") as f:
        f.write("Epoch\tTrain Loss\tValidation Loss\tValidation Accuracy\n")
        for epoch in range(epochs):
            f.write(f"{epoch+1}\t{train_losses[epoch]:.4f}\t{val_losses[epoch]:.4f}\t{val_accuracies[epoch]:.4f}\n")


if __name__ == "__main__":
    main()
