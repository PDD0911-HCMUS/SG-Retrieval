import torch
from typing import Iterable
import time
import numpy as np
from Controller.IRESGController.model.model import ModelCross
from tqdm import tqdm
import faiss

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
                    data_loader: Iterable, data_db: Iterable, 
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

        #Compute mean Recall
        compute_recall(model, data_db, device)
        

    return avg_loss

def faiss_retrieval_controller(z_que, set_z_rev, images_id_rev):
    set_z_rev = np.stack([
        t.detach().cpu().numpy() for t in set_z_rev
    ]).astype('float32')
    index = faiss.IndexFlatIP(set_z_rev.shape[1])  # Dùng Euclidean distance
    index.add(set_z_rev)
    D, I = index.search(z_que, k=50)
    selected_images = [images_id_rev[i] for i in I[0]]
    return selected_images

def compute_recall(model: ModelCross, data_db: Iterable, device):

    '''
    # image_ids_a, images_id_b: list of image_name que and rev
    # triplets_que, triplets_rev: list of triplet que and rev
    '''

    image_ids_a, images_ids_b = [], []
    triplets_que, triplets_rev = [], []
    imgs_a, imgs_b = [], []

    for img_a, img_b, trip_que, trip_rev, image_id_a, image_id_b in tqdm(data_db):
        image_ids_a.append(image_id_a[0]), images_ids_b.append(image_id_b[0])

        img_a = img_a[0].to(device)
        img_b = img_b[0].to(device)
        trip_que = [{k: v.to(device) for k, v in t.items()} for t in trip_que]
        trip_rev = [{k: v.to(device) for k, v in t.items()} for t in trip_rev]

        z_i, z_i_msk, _ = model.models.vision_encoder(img_a)
        go, _ = model.models.graph_encoder_o(trip_que)
        ge, _ = model.models.graph_encoder_e(trip_rev)

        z_o, _ = model.models.attn_graph_o(query=go,
            key=z_i,
            value=z_i,
            key_padding_mask=z_i_msk)
        
        z_e, _ = model.models.attn_graph_e(query=ge,
            key=z_i,
            value=z_i,
            key_padding_mask=z_i_msk)
        
        imgs_a.append(z_i[:,0])
        triplets_que.append(z_o[:,0][0])
        triplets_rev.append(z_e[:,0][0])

    print(f"Total list Que: {len(triplets_que)}\nSize item: {triplets_que[0].size()}")
    print(f"Total list Rev: {len(triplets_rev)}\nSize item: {triplets_rev[0].size()}")
    print(f"Total length image que: {len(image_ids_a)}\nTotal length image rev: {len(images_ids_b)}")
    x = faiss_retrieval_controller(triplets_que[0].unsqueeze(0), triplets_rev, images_ids_b)

    print(x)


        # print(z_i.size())
        # print(go.size())
        # print(ge.size())

        # print(z_o.size())
        # print(z_e.size())

        # break

    return 