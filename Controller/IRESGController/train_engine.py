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
        
        z_iA, z_iB, zt_e, z_o, z_eB_i2g = model(im_a,im_b,trip_que,trip_rev)
        losses = criterion(z_iA, z_iB, zt_e, z_o, z_eB_i2g)

        optimizer.zero_grad()
        losses['loss'].backward()

        # Gradient norm (helps control exploding gradient)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += losses['loss'].item()
        total_info_nce += losses['info_nce'].item()
        total_cosine_sim += losses['cosine_sim'].item()

        # ETA
        batch_time = time.time() - batch_start_time
        elapsed_time = time.time() - start_time
        estimated_total_time = (elapsed_time / batch_idx) * num_batches
        eta = estimated_total_time - elapsed_time

        elem_info_nce = [f"{x.item():.5f}" for x in losses['elem_info_nce']]
        elem_cosine_sim = [f"{y.item():.5f}" for y in losses['elem_cosine_sim']]
        if batch_idx % log_interval == 0 or batch_idx == num_batches:
            logger.info(
                f"Epoch {epoch} - Iter {batch_idx}/{num_batches} "
                f"- Time per batch: {batch_time:.2f}s "
                f"- ETA: {eta/60:.1f} min "
                f"- elem_info_nce = {elem_info_nce} "
                f"- elem_cosine_sim = {elem_cosine_sim} "
                f"- info_nce = {losses['info_nce'].item():.5f} "
                f"- cosine_sim = {losses['cosine_sim'].item():.5f} "
                f"- Loss = {losses['loss'].item():.5f} "
            )

            break

    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    avg_info_nce = total_info_nce / num_batches if num_batches > 0 else 0
    avg_cosine_sim = total_cosine_sim / num_batches if num_batches > 0 else 0
    logger.info(f"Epoch {epoch} - Average Training Loss: {avg_loss:.5f} "
                f"- info_nce: {avg_info_nce:.5f} "
                f"- cosine_sim: {avg_cosine_sim:.5f}")
        
    return avg_loss

def valid_engine(model: ModelCross, criterion: torch.nn.Module,
                    data_loader: Iterable, data_db: Iterable, 
                    device: torch.device, epoch: int, logger, log_interval):
    
    model.eval()
    criterion.eval()

    total_loss = 0.0
    total_info_nce = 0.0
    total_cosine_sim = 0.0
    num_batches = len(data_loader)

    start_time = time.time()
    with torch.no_grad():
        for batch_idx, (im_a, im_b, trip_que, trip_rev) in enumerate(data_loader, start=1):
            batch_start_time = time.time()
            im_a = im_a.to(device)
            im_b = im_b.to(device)
            trip_que = [{k: v.to(device) for k, v in t.items()} for t in trip_que]
            trip_rev = [{k: v.to(device) for k, v in t.items()} for t in trip_rev]

            z_iA, z_iB, zt_e, z_o, z_eB_i2g = model(im_a,im_b,trip_que,trip_rev)
            losses = criterion(z_iA, z_iB, zt_e, z_o, z_eB_i2g)
            
            total_loss += losses['loss'].item()        
            total_info_nce += losses['info_nce'].item()
            total_cosine_sim += losses['cosine_sim'].item()

            # ETA
            batch_time = time.time() - batch_start_time
            elapsed_time = time.time() - start_time
            estimated_total_time = (elapsed_time / batch_idx) * num_batches
            eta = estimated_total_time - elapsed_time

            elem_info_nce = [f"{x.item():.5f}" for x in losses['elem_info_nce']]
            elem_cosine_sim = [f"{y.item():.5f}" for y in losses['elem_cosine_sim']]
            if batch_idx % log_interval == 0 or batch_idx == num_batches:
                logger.info(
                    f"Epoch (val) {epoch} - Iter {batch_idx}/{num_batches} "
                    f"- Time per batch: {batch_time:.2f}s "
                    f"- ETA: {eta/60:.1f} min "
                    f"- elem_info_nce = {elem_info_nce} "
                    f"- elem_cosine_sim = {elem_cosine_sim} "
                    f"- info_nce = {losses['info_nce'].item():.5f} "
                    f"- cosine_sim = {losses['cosine_sim'].item():.5f} "
                    f"- Loss = {losses['loss'].item():.5f} "
                )

                break
        
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        avg_info_nce = total_info_nce / num_batches if num_batches > 0 else 0
        avg_cosine_sim = total_cosine_sim / num_batches if num_batches > 0 else 0
        logger.info(
            f"Epoch {epoch} - Validation Loss: {avg_loss:.5f} "
            f"- info_nce: {avg_info_nce:.5f} "
            f"- cosine_sim: {avg_cosine_sim:.5f}"
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
    triplets_rev = []
    images_rev = []
    # imgs_b = []
    with torch.no_grad():
        for img_a, img_b, trip_que, trip_rev, image_id_a, image_id_b in tqdm(data_db):

            images_ids_b.append(image_id_b[0])

            img_b = img_b[0].to(device)
            trip_rev = [{k: v.to(device) for k, v in t.items()} for t in trip_rev]
            z_iB, z_iB_msk, _ = model.models.vision_encoder(img_b)

            zt_e, _ = model.models.graph_encoder_e(trip_rev)
            
            z_eb, _ = model.models.attn_graph_be(
                query=zt_e,
                key=z_iB,
                value=z_iB,
                key_padding_mask=z_iB_msk
            )
            z_eb = z_eb[:,0][0]
            z_iB = z_iB[:,0][0]
            z_eb = F.normalize(z_eb, p=2, dim=1)
            z_iB = F.normalize(z_iB, p=2, dim=1)

            triplets_rev.append(z_eb)
            images_rev.append(z_iB)

        return images_ids_b, triplets_rev, images_rev

def compute_recall(model: ModelCross, data_db: Iterable, device, logger, K = [10, 20, 50]):

    '''
    # image_ids_a, images_id_b: list of image_name que and rev
    # triplets_que, triplets_rev: list of triplet que and rev
    '''

    image_ids_a = []
    # triplets_que_o, triplets_que_e = [], []
    imgs_a = []

    logger.info(f"Creating Gallery")
    images_ids_b, triplets_rev, images_rev = create_gallery(model, data_db, device)

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
            
            # imgs_a.append(z_iA[:,0])

            revO = faiss_retrieval_controller(z_o[:,0][0].unsqueeze(0), triplets_rev, images_ids_b)
            revE = faiss_retrieval_controller(ge[:,0][0].unsqueeze(0), images_rev, images_ids_b)
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