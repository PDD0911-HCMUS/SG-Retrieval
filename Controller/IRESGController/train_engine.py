import torch
from typing import Iterable
import time
import datetime
from Controller.IRESGController.model.model import ModelCross

def train_engine(model: ModelCross, criterion: torch.nn.Module,
                data_loader: Iterable, optimizer: torch.optim.Optimizer,
                device: torch.device, epoch: int, logger, log_interval):
    model.train()
    criterion.train()

    total_loss = 0.0
    total_info_nce = 0.0
    total_cosine_sim = 0.0
    num_batches = len(data_loader)

    start_time = time.time()

    for batch_idx, (im_a, im_b, trip_que, trip_rev) in enumerate(data_loader, start=1):
        batch_start_time = time.time()
        im_a = im_a.to(device)
        im_b = im_b.to(device)
        trip_que = [{k: v.to(device) for k, v in t.items()} for t in trip_que]
        trip_rev = [{k: v.to(device) for k, v in t.items()} for t in trip_rev]

        # print(trip_que)
        
        out_o, out_e, out_i, out_be = model(im_a,im_b,trip_que,trip_rev)
        losses = criterion(out_i, out_o, out_e, out_be)

        optimizer.zero_grad()
        losses['loss'].backward()

        # Gradient norm (helps control exploding gradient)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += losses['loss'].item()
        total_info_nce += losses['info_nce'].item()
        total_cosine_sim += losses['avg_cosine'].item()

        # ETA
        batch_time = time.time() - batch_start_time
        elapsed_time = time.time() - start_time
        estimated_total_time = (elapsed_time / batch_idx) * num_batches
        eta = estimated_total_time - elapsed_time

        # Logging loss
        if batch_idx % log_interval == 0 or batch_idx == num_batches:
            logger.info(
                f"Epoch {epoch} - Iter {batch_idx}/{num_batches} "
                f"- Time per batch: {batch_time:.2f}s "
                f"- ETA: {eta/60:.1f} min "
                f"- info_nce = {losses['info_nce'].item():.4f} "
                f"- cosine_sim = {losses['cosine_sim'].item():.4f} "
                f"- Loss = {losses['loss'].item():.4f} "
                f"- Grad Norm: {grad_norm:.4f}"
            )

        break

    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    avg_info_nce = total_info_nce / num_batches if num_batches > 0 else 0
    avg_cosine_sim = total_cosine_sim / num_batches if num_batches > 0 else 0
    logger.info(f"Epoch {epoch} - Average Training Loss: {avg_loss}"
                f"- info_nce: {avg_info_nce} "
                f"- cosine_sim: {avg_cosine_sim}")
        
    return avg_loss

def valid_engine(model: ModelCross, criterion: torch.nn.Module,
                    data_loader: Iterable,
                    device: torch.device, epoch: int, logger):
    
    model.eval()
    criterion.eval()

    # ve = model.models.vision_encoder()

    total_loss = 0.0
    total_info_nce = 0.0
    total_cosine_sim = 0.0
    num_batches = len(data_loader)

    with torch.no_grad():
        for batch_idx, (im_a, im_b, trip_que, trip_rev) in enumerate(data_loader, start=1):
            im_a = im_a.to(device)
            im_b = im_b.to(device)
            trip_que = [{k: v.to(device) for k, v in t.items()} for t in trip_que]
            trip_rev = [{k: v.to(device) for k, v in t.items()} for t in trip_rev]

            out_a, out_r_a, out_b, out_r_b = model(im_a,im_b,trip_que,trip_rev)
            losses = criterion(out_a, out_r_a, out_b, out_r_b)
            
            total_loss += losses['loss'].item()        
            total_info_nce += losses['info_nce'].item()
            total_cosine_sim += losses['avg_cosine'].item()

            break

    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    avg_info_nce = total_info_nce / num_batches if num_batches > 0 else 0
    avg_cosine_sim = total_cosine_sim / num_batches if num_batches > 0 else 0
    logger.info(
        f"Epoch {epoch} - Validation Loss: {avg_loss} "
        f"- info_nce: {avg_info_nce} "
        f"- cosine_sim: {avg_cosine_sim}"
    )
    return avg_loss

def compute_recall(model, val_que, index, k=[1,5,10]):

    return 