from config import db
from flask import request, jsonify, send_from_directory
from sqlalchemy.exc import SQLAlchemyError
import torch
from pathlib import Path
import numpy as np
from Controller.IRESGController.model.model_v2 import ModelCross
from torch.utils.data import DataLoader, SequentialSampler
from Controller.IRESGController.model.model_v2 import build, ModelCross
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

class IRESGController:
    def __init__(self):
        self.ckpt = ckpt
        self.hidden_dim = hidden_dim
        self.lr_backbone = lr_backbone
        self.masks = masks
        self.backbone = backbone
        self.dilation = dilation
        self.nhead = nhead
        self.nlayer = nlayer
        self.d_ffn = d_ffn
        self.dropout = dropout
        self.random_erasing_prob = random_erasing_prob
        self.activation = activation
        self.pre_train = pre_train
        self.device = device
        self.transform = T.Compose([
            T.Resize(512),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.dir_upload = args.ConfigData.dir_upload

        self.image_folder = vg_image_dir
        self.tokenizer = tokenizer
        self.max_lenght = max_lenght
        self.anno = anno
        self.num_workers = num_workers
    
    @staticmethod
    def faiss_retrieval_controller(z_que, set_z_rev, images_id_rev):
        # z_que = F.normalize(z_que, p=2, dim=1)
        if isinstance(z_que, torch.Tensor):
            z_que = z_que.detach().cpu().numpy().astype('float32')
        set_z_rev = np.stack(set_z_rev).astype('float32')
        print(set_z_rev.shape)
        index = faiss.IndexFlatIP(set_z_rev.shape[1])  # Dùng Euclidean distance
        index.add(set_z_rev)
        D, I = index.search(z_que, k=50)
        selected_images = [images_id_rev[i] for i in I[0]]
        return selected_images
    
    @staticmethod
    def get_embedding_v2(model: ModelCross, img_id, img, device):
        with torch.no_grad():
            img = img[0].to(device)

            z_i, z_i_msk, _ = model.models.vision_encoder(img)

            return img_id[0], z_i[:,0]

    @staticmethod
    def get_embedding_query(model: ModelCross, img, triplet, mode, device):
        with torch.no_grad():
            img = img.to(device)
            triplet = [{k: v.to(device) for k, v in t.items()} for t in triplet]

            z_i, z_i_msk, _ = model.models.vision_encoder(img)
            z_t, _ = model.models.graph_encoder_o(triplet) 

            if(mode == 0):
                print("RUN MODE 0000000000000000000000")
                
                z_cross, _ = model.models.attn_graph_o(
                    query=z_t,
                    key=z_i,
                    value=z_i,
                    key_padding_mask=z_i_msk
                )
                z_que = F.normalize(z_cross, p=2, dim=1)
            if(mode == 1):
                print("RUN MODE 111111111111111111111")
                z_t, _ = model.models.graph_encoder_e(triplet)
                z_que = F.normalize(z_t, p=2, dim=1)

            # return z_cross[:,0][0], z_i[:,0], z_t[:,0]
            return z_que[:,0][0]
        
    @staticmethod
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
    
    def _get_model(self):
        model, _ = build(
            self.hidden_dim,
            self.lr_backbone,
            self.masks, 
            self.backbone, 
            self.dilation,
            self.nhead, 
            self.nlayer, 
            self.d_ffn, 
            self.dropout, 
            self.random_erasing_prob, 
            self.activation, 
            self.pre_train
            )
        
        checkpoint = torch.load(
            self.ckpt, 
            map_location=torch.device(self.device)
            )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        model = model.to(self.device)
        return model
    
    def _create_input(self, image, triplet):
        
        tokenizer = BertTokenizer.from_pretrained(self.pre_train)
        print(triplet)
        img = self.transform(image).unsqueeze(0)
        triplet_enc = tokenizer(triplet, padding = 'max_length', truncation = True, max_length = self.max_lenght, return_tensors = 'pt')
        triplet = {
                'trip_ids': triplet_enc['input_ids'],         # shape: [num_triplets, max_len]
                'trip_mask': triplet_enc['attention_mask'] # shape: [num_triplets, max_len]
            }
        trip = IRESGController.pad_or_truncate_tensor(triplet)

        return img, trip
    
    def retrieve(self):
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
                filepath = os.path.join(self.dir_upload, file.filename)
                file.save(filepath)

            fileName = file.filename
            file_name = self.dir_upload + fileName
            path = Path(file_name.replace('.jpg', '').replace('.png', ''))
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)
            
            image = Image.open(file_name)
            triplet = json.loads(triplet_str)

            img, trip = self._create_input(image, triplet)

            model = self._get_model()

            z_que = self.get_embedding_query(model, img, [trip], edit, device)
        
            z_que = z_que.unsqueeze(0)

            IRESGVGV2 = entity.IRESGVGV2
            # IRESGMSCOCOV2 = entity.IRESGMSCOCOV2
            image_ids = []
            embeddings = []
            gallery = db.session.query(
                # IRESGMSCOCOV2.image_id,
                # IRESGMSCOCOV2.embedding

                IRESGVGV2.image_id,
                IRESGVGV2.embedding
            ).all()

            for image_id, embedding in gallery:
                image_ids.append(image_id)
                embeddings.append(np.array(embedding[0], dtype=np.float32))

            selected_images = self.faiss_retrieval_controller(
                images_id_rev=image_ids,
                set_z_rev=embeddings,
                z_que=z_que
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
        
    def create_gallery(self):
        dataset_db = create_db(
            image_folder=self.image_folder,
            ann_file=self.anno,
            tokenizer=self.tokenizer,
            max_length=self.max_lenght
        )

        sampler_db = SequentialSampler(dataset_db)
        data_db = DataLoader(dataset_db,
                batch_size=1, 
                sampler=sampler_db,
                drop_last=False,
                collate_fn=collate_fn_dual_image_db,
                num_workers=self.num_workers,
                pin_memory=True)

        model = self._get_model()

        IRESGVGV2 = entity.IRESGVGV2
        IRESGMSCOCOV2 = entity.IRESGMSCOCOV2
        try:    
            for img_a, img_b, trip_que, trip_rev, image_id_a, image_id_b in tqdm(data_db):

                im_id_o, z_i_o = self.get_embedding_v2(model, image_id_a, img_a,device)
                im_id_e, z_i_e = self.get_embedding_v2(model, image_id_b, img_b,device)

                insert_o = IRESGMSCOCOV2(
                    image_id = im_id_o,
                    embedding = z_i_o.tolist(),
                    # triplets = trip_que
                )

                insert_e = IRESGMSCOCOV2(
                    image_id = im_id_e,
                    embedding = z_i_e.tolist(),
                    # triplets = trip_rev
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

    def serve_image(self, filename):
        return send_from_directory(self.image_folder, filename)