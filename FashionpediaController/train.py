from datasets.data_pre import build_data
from model_cross import build_model
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
from tqdm import tqdm

def main():

    num_epochs = 1
    ckpt = 'ckpt/'
    saved_model_epoch = 'cross_modal_model_with_attention_epoch_'
    saved_model = 'cross_modal_model_with_attention'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_train = build_data('train')
    data_valid = build_data('val')

    index = 5
    input_ids, attention_mask, label_ids, label_attention_mask = data_train.__getitem__(index)

    dataloader_train = DataLoader(data_train, batch_size=4, shuffle=True)
    dataloader_valid = DataLoader(data_valid, batch_size=4, shuffle=True)

    #check dataset
    for i, (input_ids, attention_mask, label_ids, label_attention_mask) in enumerate(dataloader_train):
        print(f"Batch {i+1}")
        print(f"Input IDs: {input_ids.size()}")
        print(f"Attention Mask: {attention_mask.size()}")
        print(f"Label IDs: {label_ids.size()}")
        print(f"Label Attention Mask: {label_attention_mask.size()}")
        print("\n")
        if i == 1:
            break

    model, criterion = build_model(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    with open(f'{ckpt}training_log.txt', 'w') as log_file:
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0
            print(f'training epoch {epoch}/{num_epochs}: ')
            for input_ids, attention_mask, label_ids, label_attention_mask in tqdm(dataloader_train):
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                label_ids = label_ids.to(device)
                label_attention_mask = label_attention_mask.to(device)

                optimizer.zero_grad()
                text_features, label_features = model(input_ids, attention_mask, label_ids, label_attention_mask)
                loss = criterion(text_features, label_features)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                
            train_loss /= len(dataloader_train)
            log_file.write(f'Epoch {epoch+1}, Train Loss: {train_loss}\n')
            print(f'Epoch {epoch+1}, Train Loss: {train_loss}')
            
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                print(f'Validation epoch {epoch}/{num_epochs}: ')
                for input_ids, attention_mask, label_ids, label_attention_mask in tqdm(dataloader_valid):
                    input_ids = input_ids.to(device)
                    attention_mask = attention_mask.to(device)
                    label_ids = label_ids.to(device)
                    label_attention_mask = label_attention_mask.to(device)

                    text_features, label_features = model(input_ids, attention_mask, label_ids, label_attention_mask)
                    loss = criterion(text_features, label_features)
                    val_loss += loss.item()
            
            val_loss /= len(dataloader_valid)
            log_file.write(f'Epoch {epoch+1}, Validation Loss: {val_loss}\n')
            torch.save(model.state_dict(), f'{ckpt}{saved_model_epoch}_{epoch+1}.pth')

        torch.save(model.state_dict(), f'{ckpt}{saved_model}.pth')

if __name__ == main():
    main()