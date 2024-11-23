from Dataset.create_data import build_data
from .create_model import build_model
import torch
from torch.utils.data import DataLoader
import datetime
import time
import torch.optim as optim
from tqdm import tqdm
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    #For Dataset
    batch_size = 32
    ratio = 0.8
    csv_file = '/radish/phamd/duypd-proj/SG-Retrieval/khangTA/Dataset/final_data.csv'

    #For model
    pretrained = 'vinai/phobert-base'
    num_classes = 9
    drop_out = 0.1
    start_epoch = 0
    epochs = 50
    learning_rate = 1e-4
    step_size = 10
    gamma = 0.1   

    #====================================================#

    print('Preparing Dataset')
    train_dataset, valid_dataset = build_data(csv_file, ratio)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    print("Data Summary:")
    print(f"Total samples in train set: {len(train_dataset)}")
    print(f"Total samples in validation set: {len(valid_dataset)}")
    print(f"Batch size: {batch_size}")
    print(f"Total train batches: {len(train_loader)}")
    print(f"Total validation batches: {len(valid_loader)}")

    #===================================================#

    print('Preparing Model')
    model, criterion = build_model(from_pretrained=pretrained,
                                   num_classes=num_classes,
                                   drop_out=drop_out)
    
    model.to(device)
    criterion.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    print(f"\nTotal trainable parameters: {count_parameters(model)}")

    #===================================================#
    train_losses = []
    val_losses = []
    val_accuracies = []

    print("Start training")
    start_time = time.time()
    for epoch in range(start_epoch, epochs):
        print(f"Training Epoch [{epoch+1}/{epochs}]")
        model.train()
        criterion.train()
        total_train_loss = 0

        for samples, msks, labels in tqdm(train_loader):
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
            print(f"Validation Epoch [{epoch+1}/{epochs}]")
            for samples, msks, labels in tqdm(valid_loader):
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
            torch.save(model.state_dict(), f"/radish/phamd/duypd-proj/SG-Retrieval/khangTA/ClassificationController/ckpt/model_epoch_{epoch+1}.pth")

        print(f"Epoch [{epoch+1}/{epochs}], "
          f"Train Loss: {avg_train_loss:.4f}, "
          f"Validation Loss: {avg_val_loss:.4f}, "
          f"Validation Accuracy: {val_accuracy:.4f}, "
          f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
    with open("training_log.txt", "w") as f:
        f.write("Epoch\tTrain Loss\tValidation Loss\tValidation Accuracy\n")
        for epoch in range(epochs):
            f.write(f"{epoch+1}\t{train_losses[epoch]:.4f}\t{val_losses[epoch]:.4f}\t{val_accuracies[epoch]:.4f}\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == "__main__":
    main()