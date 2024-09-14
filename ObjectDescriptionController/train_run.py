from .datasets.data import build_data, vocab, collate_fn
from .model.obde import build_model
from torch.utils.data import DataLoader
import datetime
import time
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch
from nltk.translate.bleu_score import corpus_bleu
import os
import json

def greedy_decode(model, image, max_len, vocab):
    model.eval()
    with torch.no_grad():
        feature = model.encoder(image.unsqueeze(0))  # Encode image
        output = [vocab.stoi["<SOS>"]]  # Start with <SOS>

        for _ in range(max_len):
            output_tensor = torch.tensor(output).unsqueeze(0).to(image.device)
            preds = model.decoder(feature, output_tensor)
            preds = preds.squeeze(0)
            next_token = preds.argmax(1)[-1].item()  # Get highest probability token

            if next_token == vocab.stoi["<EOS>"]:
                break
            output.append(next_token)
        
        predicted_caption = [vocab.itos[idx] for idx in output if idx not in {vocab.stoi["<SOS>"], vocab.stoi["<EOS>"], vocab.stoi["<PAD>"]}]
    return predicted_caption

def calculate_bleu(references, hypotheses):
    bleu1 = corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0))  # BLEU-1
    bleu2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0))  # BLEU-2
    bleu3 = corpus_bleu(references, hypotheses, weights=(0.33, 0.33, 0.33, 0))  # BLEU-3
    bleu4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25))  # BLEU-4
    return bleu1, bleu2, bleu3, bleu4

def validate_and_evaluate(model, data_val, criterion, vocab, device, max_len=20):
    model.eval()
    total_loss = 0
    references = []
    hypotheses = []

    with torch.no_grad():
        for images, captions in data_val:
            images, captions = images.to(device), captions.to(device)
            outputs = model(images, captions[:, :-1])
            loss = criterion(outputs, captions[:, 1:])
            total_loss += loss.item()

            for i in range(images.size(0)):
                predicted_caption = greedy_decode(model, images[i], max_len, vocab)
                target_caption = [vocab.itos[idx] for idx in captions[i].cpu().numpy() if idx not in {vocab.stoi["<SOS>"], vocab.stoi["<EOS>"], vocab.stoi["<PAD>"]}]
                
                hypotheses.append(predicted_caption)
                references.append([target_caption])  # BLEU yêu cầu reference là list of lists

    avg_loss = total_loss / len(data_val)
    bleu1, bleu2, bleu3, bleu4 = calculate_bleu(references, hypotheses)

    print(f"Validation Loss: {avg_loss:.4f}, BLEU-1: {bleu1:.4f}, BLEU-2: {bleu2:.4f}, BLEU-3: {bleu3:.4f}, BLEU-4: {bleu4:.4f}")
    return avg_loss, bleu1, bleu2, bleu3, bleu4

def train_run(model, criterion, data_train, data_val, vocab, num_epochs, device, optimizer, scheduler, save_dir):

    metrics = {
        "train_loss": [],
        "val_loss": [],
        "bleu1": [],
        "bleu2": [],
        "bleu3": [],
        "bleu4": [],
        "learning_rate": []
    }

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for images, captions in data_train:
            images, captions = images.to(device), captions.to(device)

            outputs = model(images, captions[:, :-1])
            loss = criterion(outputs, captions[:, 1:])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        scheduler.step()
        avg_train_loss = total_loss / len(data_train)
        avg_val_loss, bleu1, bleu2, bleu3, bleu4 = validate_and_evaluate(model, data_val, criterion, vocab, device)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")
        print(f"BLEU-1: {bleu1:.4f}, BLEU-2: {bleu2:.4f}, BLEU-3: {bleu3:.4f}, BLEU-4: {bleu4:.4f}")

        metrics["train_loss"].append(avg_train_loss)
        metrics["val_loss"].append(avg_val_loss)
        metrics["bleu1"].append(bleu1)
        metrics["bleu2"].append(bleu2)
        metrics["bleu3"].append(bleu3)
        metrics["bleu4"].append(bleu4)
        metrics["learning_rate"].append(optimizer.param_groups[0]["lr"])

        if(epoch % 2 == 0):
            model_save_path = os.path.join(save_dir, f"model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), model_save_path)
        
    metrics_save_path = os.path.join(save_dir, "training_metrics.json")
    with open(metrics_save_path, 'w') as f:
        json.dump(metrics, f)
    print(f"Training metrics saved to {metrics_save_path}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 32

    embed_size = 256
    vocab_size = len(vocab)
    num_heads = 8
    hidden_dim = 512
    num_layers = 6
    pad_idx = vocab.stoi["<PAD>"]

    lr = 0.001
    weight_decay = 1e-5
    gamma  = 0.1
    step_size = 5
    num_epochs = 30
    save_dir = '/home/duypd/ThisPC-DuyPC/SG-Retrieval/ckpt/obde/'
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    train_ds, val_ds = build_data()

    train_loader = DataLoader(
        dataset=train_ds, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4, 
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        dataset=val_ds, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=4, 
        collate_fn=collate_fn
    )

    model, criterion = build_model(
        embed_size=embed_size,
        vocab_size=vocab_size,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        pad_idx=pad_idx,
        device = device
    )

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    # print("Start training")
    # start_time = time.time()
    # train_run(model, criterion, train_loader, val_loader, vocab, num_epochs, device, optimizer, scheduler, save_dir)
    # total_time = time.time() - start_time
    # total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    # print('Training time {}'.format(total_time_str))

if __name__ == "__main__":
    main()