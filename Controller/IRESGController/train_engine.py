import torch
from typing import Iterable
import time
import numpy as np
from Controller.IRESGController.model.model_v2 import ModelCross
from tqdm import tqdm
import faiss
from collections import defaultdict
import torch.nn.functional as F

def train_engine(model: ModelCross, criterion: torch.nn.Module,
                data_loader: Iterable, optimizer: torch.optim.Optimizer,
                device: torch.device, epoch: int, logger, log_interval):
    model.train()
    criterion.train()

    total_loss = 0.0
    total_sim = 0.0
    num_batches = len(data_loader)

    start_time = time.time()

    for batch_idx, (im_a, im_b, trip_que, trip_rev) in enumerate(data_loader, start=1):
        batch_start_time = time.time()
        im_a = im_a.to(device)
        im_b = im_b.to(device)
        trip_que = [{k: v.to(device) for k, v in t.items()} for t in trip_que]
        trip_rev = [{k: v.to(device) for k, v in t.items()} for t in trip_rev]
        
        z_iA, z_iB, zt_e, z_o = model(im_a,im_b,trip_que,trip_rev)
        losses = criterion(z_iA, z_iB, zt_e, z_o)

        optimizer.zero_grad()
        losses['loss'].backward()

        # Gradient norm (helps control exploding gradient)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += losses['loss'].item()

        total_sim += losses['sim'].item()

        # ETA
        batch_time = time.time() - batch_start_time
        elapsed_time = time.time() - start_time
        estimated_total_time = (elapsed_time / batch_idx) * num_batches
        eta = estimated_total_time - elapsed_time

        elem_sim = [f"{y.item():.5f}" for y in losses['elem_sim']]
        if batch_idx % log_interval == 0 or batch_idx == num_batches:
            logger.info(
                f"Epoch {epoch} - Iter {batch_idx}/{num_batches} "
                f"- Time per batch: {batch_time:.2f}s "
                f"- ETA: {eta/60:.1f} min "
                f"- elem_sim = {elem_sim} "
                f"- sim = {losses['sim'].item():.5f} "
                f"- Loss = {losses['loss'].item():.5f} "
            )

            # break

    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    avg_sim = total_sim / num_batches if num_batches > 0 else 0
    logger.info(f"Epoch {epoch} - Average Training Loss: {avg_loss:.5f} "
                f"- sim: {avg_sim:.5f}")
        
    return avg_loss

def valid_engine(model: ModelCross, criterion: torch.nn.Module,
                    data_loader: Iterable, data_db: Iterable, 
                    device: torch.device, epoch: int, logger, log_interval):
    
    model.eval()
    criterion.eval()

    total_loss = 0.0

    total_sim = 0.0
    num_batches = len(data_loader)

    start_time = time.time()
    with torch.no_grad():
        for batch_idx, (im_a, im_b, trip_que, trip_rev) in enumerate(data_loader, start=1):
            batch_start_time = time.time()
            im_a = im_a.to(device)
            im_b = im_b.to(device)
            trip_que = [{k: v.to(device) for k, v in t.items()} for t in trip_que]
            trip_rev = [{k: v.to(device) for k, v in t.items()} for t in trip_rev]

            z_iA, z_iB, zt_e, z_o = model(im_a,im_b,trip_que,trip_rev)
            losses = criterion(z_iA, z_iB, zt_e, z_o)
            
            total_loss += losses['loss'].item()        

            total_sim += losses['sim'].item()

            # ETA
            batch_time = time.time() - batch_start_time
            elapsed_time = time.time() - start_time
            estimated_total_time = (elapsed_time / batch_idx) * num_batches
            eta = estimated_total_time - elapsed_time

            elem_sim = [f"{y.item():.5f}" for y in losses['elem_sim']]
            if batch_idx % log_interval == 0 or batch_idx == num_batches:
                logger.info(
                    f"Epoch (val) {epoch} - Iter {batch_idx}/{num_batches} "
                    f"- Time per batch: {batch_time:.2f}s "
                    f"- ETA: {eta/60:.1f} min "
                    f"- elem_sim = {elem_sim} "
                    f"- sim = {losses['sim'].item():.5f} "
                    f"- Loss = {losses['loss'].item():.5f} "
                )

                # break
        
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        avg_sim = total_sim / num_batches if num_batches > 0 else 0
        logger.info(
            f"Epoch {epoch} - Validation Loss: {avg_loss:.5f} "
            f"- sim: {avg_sim:.5f}"
        )

        #Compute mean Recall
        compute_recall(model, data_db, device, logger)
        
    return avg_loss

def faiss_retrieval_controller(z_que, set_z_rev, images_id_rev):
    z_que = F.normalize(z_que, p=2, dim=1)
    if isinstance(z_que, torch.Tensor):
        z_que = z_que.detach().cpu().numpy().astype('float32')
    set_z_rev = np.stack([
        t.detach().cpu().numpy() for t in set_z_rev
    ]).astype('float32')
    index = faiss.IndexFlatIP(set_z_rev.shape[1])  # Dùng Euclidean distance
    index.add(set_z_rev)
    D, I = index.search(z_que, k=50)
    selected_images = [images_id_rev[i] for i in I[0]]
    return selected_images

def create_gallery(model: ModelCross, data_db: Iterable, device):
    images_ids_b = []
    images_rev = []
    # imgs_b = []
    with torch.no_grad():
        for img_a, img_b, trip_que, trip_rev, image_id_a, image_id_b in tqdm(data_db):

            images_ids_b.append(image_id_b[0])

            img_b = img_b[0].to(device)

            z_iB, z_iB_msk, _ = model.models.vision_encoder(img_b)
            
            z_iB = model.models.proj(z_iB[:,0])

            z_iB = F.normalize(z_iB, p=2, dim=1)

            images_rev.append(z_iB[0])

        return images_ids_b, images_rev

def compute_recall(model: ModelCross, data_db: Iterable, device, logger, K = [10, 20, 50]):

    image_ids_a = []

    logger.info(f"Creating Gallery")
    images_ids_b, images_rev = create_gallery(model, data_db, device)

    hits_o = defaultdict(int)
    hits_e = defaultdict(int)

    logger.info(f"Start Running Validation")
    with torch.no_grad():
        for img_a, img_b, trip_que, trip_rev, image_id_a, image_id_b in tqdm(data_db):
            image_ids_a.append(image_id_a[0])

            img_a = img_a[0].to(device)
            trip_que = [{k: v.to(device) for k, v in t.items()} for t in trip_que]
            trip_rev = [{k: v.to(device) for k, v in t.items()} for t in trip_rev]

            z_iA, z_iA_msk, _ = model.models.vision_encoder(img_a)

            go, _ = model.models.graph_encoder_o(trip_que)
            ge, _ = model.models.graph_encoder_e(trip_rev)

            z_o, _ = model.models.attn_graph_o(
                query=go,
                key=z_iA,
                value=z_iA,
                key_padding_mask=z_iA_msk)

            z_o = model.models.proj(z_o[:, 0])
            ge = model.models.proj(ge[:, 0])

            revO = faiss_retrieval_controller(z_o, images_rev, images_ids_b)
            revE = faiss_retrieval_controller(ge, images_rev, images_ids_b)
            # print(image_id_a[0], image_id_b[0])
            for k in K:
                if(image_id_b[0] in revO[:k]):
                    hits_o[k] += 1
                if(image_id_b[0] in revE[:k]):
                    hits_e[k] += 1

            # break

        recall_o = {k: hits_o[k] / len(data_db) for k in K}
        recall_e = {k: hits_e[k] / len(data_db) for k in K}
        # print("Recall@K for z_o:", recall_o)
        # print("Recall@K for z_e:", recall_e)
        logger.info(f"========== Recall for non-Editted and Editted ==========")
        logger.info(f"non-Editted | R@10: {recall_o[10]:.5f} | R@20: {recall_o[20]:.5f} | R@50: {recall_o[50]:.5f}")
        logger.info(f"Editted     | R@10: {recall_e[10]:.5f} | R@20: {recall_e[20]:.5f} | R@50: {recall_e[50]:.5f}")

    return 