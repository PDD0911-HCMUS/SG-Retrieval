from config import db
from flask_cors import CORS, cross_origin
from flask import Blueprint, request, jsonify, send_from_directory
from sqlalchemy.exc import SQLAlchemyError
import torch
from pathlib import Path
import numpy as np
from Controller.IRESGController.model.model import ModelCross
from torch.utils.data import DataLoader, SequentialSampler
from Controller.IRESGController.model.model import build, ModelCross
from tqdm import tqdm
import faiss
import torch.nn.functional as F
import torchvision.transforms as T
from Controller.IRESGController.config_run import *
from Controller.IRESGController.dataset.create_db import create_db, collate_fn_dual_image_db
import Entities.entities as entity
from PIL import Image
import json
from transformers import BertTokenizer
import traceback

rev_v2_api = Blueprint('rev_v2', __name__)

def get_model():
    model, _ = build(hidden_dim,lr_backbone,masks, backbone, dilation, 
                nhead, nlayer, d_ffn, dropout, random_erasing_prob, activation, pre_train)
    
    checkpoint = torch.load(ckpt, map_location=torch.device(device))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model = model.to(device)
    return model

def faiss_retrieval_controller(z_que, set_z_rev, images_id_rev):
    # z_que = F.normalize(z_que, p=2, dim=1)
    if isinstance(z_que, torch.Tensor):
        z_que = z_que.detach().cpu().numpy().astype('float32')
    set_z_rev = np.stack(set_z_rev).astype('float32')
    index = faiss.IndexFlatIP(set_z_rev.shape[1])  # Dùng Euclidean distance
    index.add(set_z_rev)
    D, I = index.search(z_que, k=50)
    selected_images = [images_id_rev[i] for i in I[0]]
    return selected_images

def get_embedding(model: ModelCross, img_id, img, triplet, mode, device):
    with torch.no_grad():
        img = img[0].to(device)
        triplet = [{k: v.to(device) for k, v in t.items()} for t in triplet]

        z_i, z_i_msk, _ = model.models.vision_encoder(img)

        if(mode == 0):
            go, _ = model.models.graph_encoder_o(triplet) 
            z_cross, _ = model.models.attn_graph_o(
                query=go,
                key=z_i,
                value=z_i,
                key_padding_mask=z_i_msk
            )
            z_cross = F.normalize(z_cross, p=2, dim=1)
        if(mode == 1):
            ge, _ = model.models.graph_encoder_e(triplet)
            z_cross, _ = model.models.attn_graph_be(
                query=ge,
                key=z_i,
                value=z_i,
                key_padding_mask=z_i_msk
            )
            z_cross = F.normalize(z_cross, p=2, dim=1)

        return img_id[0], z_cross[:,0]
    
def get_embedding_query(model: ModelCross, img, triplet, mode, device):
    with torch.no_grad():
        img = img.to(device)
        triplet = [{k: v.to(device) for k, v in t.items()} for t in triplet]

        z_i, z_i_msk, _ = model.models.vision_encoder(img)

        if(mode == 0):
            print("RUN MODE 0000000000000000000000")
            go, _ = model.models.graph_encoder_o(triplet) 
            z_cross, _ = model.models.attn_graph_o(
                query=go,
                key=z_i,
                value=z_i,
                key_padding_mask=z_i_msk
            )
            z_cross = F.normalize(z_cross, p=2, dim=1)
        if(mode == 1):
            print("RUN MODE 111111111111111111111")
            ge, _ = model.models.graph_encoder_e(triplet)
            z_cross, _ = model.models.attn_graph_be(
                query=ge,
                key=z_i,
                value=z_i,
                key_padding_mask=z_i_msk
            )
            z_cross = F.normalize(z_cross, p=2, dim=1)

        return z_cross[:,0][0]

def pad_or_truncate_tensor(item):
    for key in ['trip_ids', 'trip_mask']:
        if key in item:
            seq = item[key]
            if seq.size(0) < max_triplet:
                padding = torch.zeros((max_triplet - seq.size(0), seq.size(1)), dtype=seq.dtype)
                item[key] = torch.cat([seq, padding], dim=0)
            elif seq.size(0) > max_triplet:
                item[key] = seq[:max_triplet]
    return item

def create_input(image, triplet):
    transform = T.Compose([
        T.Resize(512),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    tokenizer = BertTokenizer.from_pretrained(pre_train)
    print(triplet)
    img = transform(image).unsqueeze(0)
    triplet_enc = tokenizer(triplet, padding = 'max_length', truncation = True, max_length = max_lenght, return_tensors = 'pt')
    triplet = {
            'trip_ids': triplet_enc['input_ids'],         # shape: [num_triplets, max_len]
            'trip_mask': triplet_enc['attention_mask'] # shape: [num_triplets, max_len]
        }
    trip = pad_or_truncate_tensor(triplet)

    return img, trip

@rev_v2_api.route('/create_gallery', methods = ['GET'])
@cross_origin()
def create_gallery():
    dataset_db = create_db(
        image_folder=vg_image_dir,
        ann_file=anno,
        tokenizer=tokenizer,
        max_length=max_lenght
    )

    sampler_db = SequentialSampler(dataset_db)
    data_db = DataLoader(dataset_db,
            batch_size=1, 
            sampler=sampler_db,
            drop_last=False,
            collate_fn=collate_fn_dual_image_db,
            num_workers=num_workers,
            pin_memory=True)

    model = get_model()

    IRESGVG = entity.IRESGVG
    try:    
        for img_a, img_b, trip_que, trip_rev, image_id_a, image_id_b in tqdm(data_db):

            im_id_o, z_cross_o = get_embedding(model, image_id_a, img_a, trip_que, 0, device)
            im_id_e, z_cross_e = get_embedding(model, image_id_b, img_b, trip_rev, 1, device)

            insert_o = IRESGVG(
                image_id = im_id_o,
                cross_embedding = z_cross_o.tolist()
            )

            insert_e = IRESGVG(
                image_id = im_id_e,
                cross_embedding = z_cross_e.tolist()
            )

            db.session.add(insert_o)
            db.session.add(insert_e)
            # break
        db.session.commit()
        return jsonify(
            Data = "",
            Status = True, 
            Msg = 'OK'
        )
    except SQLAlchemyError as e:
        print(str(e))
        db.session.rollback()
        return jsonify(
            Data = None,
            Status = False, 
            Msg = f'Error: {e}'
        )
    finally:
        db.session.close()

@rev_v2_api.route('/retrieve', methods = ['POST'])
@cross_origin()
def retrieve():
    try:
        if 'file' not in request.files:
            return jsonify(
                Data = None,
                Status = False, 
                Msg = 'No file part in the request'
                )

        file = request.files['file']
        triplet_str = request.form['triplets']
        edit = int(request.form['edit'])
        if file.filename == '':
            return jsonify(
                Data = None,
                Status = False, 
                Msg = 'No selected file'
            )
        if file:
            filepath = os.path.join(args.ConfigData.dir_upload, file.filename)
            file.save(filepath)

        fileName = file.filename
        file_name = args.ConfigData.dir_upload + fileName
        path = Path(file_name.replace('.jpg', '').replace('.png', ''))
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
        
        image = Image.open(file_name)
        triplet = json.loads(triplet_str)

        img, trip = create_input(image, triplet)

        model = get_model()

        z_cross = get_embedding_query(model, img, [trip], edit, device)
        z_cross = z_cross.unsqueeze(0)

        IRESGVG = entity.IRESGVG
        image_ids = []
        embeddings = []
        gallery = db.session.query(
            IRESGVG.image_id,
            IRESGVG.cross_embedding
        ).all()

        for image_id, embedding in gallery:
            image_ids.append(image_id)
            embeddings.append(np.array(embedding, dtype=np.float32))

        selected_images = faiss_retrieval_controller(
            images_id_rev=image_ids,
            set_z_rev=embeddings,
            z_que=z_cross
        )

        res = {
            "imgs": selected_images,
            # "dist": dist[0].tolist(),
            "triplets": None
        }

        return jsonify(
            Data = res,
            Status = True, 
            Msg = ''
        )

    except Exception as e:
        return jsonify(
            Data = None,
            Status = False, 
            Msg = traceback.format_exc()
        )