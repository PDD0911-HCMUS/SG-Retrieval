from QueryController.datasets.data_pre import build_data, tokenize_triplets
from QueryController.model_cross import build_model
from pycocotools.coco import COCO
import torch.nn.functional as F
import torch
import torch.nn as nn
from tqdm import tqdm
import json
import psycopg2
import ConfigArgs as args
from flask_cors import CORS, cross_origin
from flask import Blueprint, request, jsonify, send_from_directory

rev_api = Blueprint('rev', __name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_postgres_array(data):
    cleaned_data = data.replace('{', '').replace('}', '').replace('""', '"')
    list_data = cleaned_data.split('","')
    list_data = [item.strip('"') for item in list_data]
    
    return list_data

def extract_subject_relation_object(trips):
    trip = []
    for item in trips:
        words = item.split()
        t_json = {
            "subject": words[0],
            "relation": ' '.join(words[1:-1]),
            "object": words[-1]
        }
        trip.append(t_json)
    return trip

def compute_text_embedding(text_triplets, model, device):
    
    input_ids, attention_mask = tokenize_triplets(text_triplets)
    
    
    input_ids = input_ids.unsqueeze(0).to(device)
    attention_mask = attention_mask.unsqueeze(0).to(device)
    
    
    with torch.no_grad():
        # text_features, _ = model.text_encoder(input_ids, attention_mask)
        text_features = model.text_encoder(input_ids, attention_mask)
    
    return text_features

def compute_label_embeddings(triplets_list, model, device):
    try:

        # connect to DB
        conn = psycopg2.connect(args.conn_str)
        cursor = conn.cursor()

        with open('Datasets/VisualGenome/rel.json', 'r') as f:
            all_rels = json.load(f)

        with open('Datasets/VisualGenome/categories.json', 'r') as f:
            categories = json.load(f)

        coco = COCO('Datasets/VisualGenome/train.json')
        rel_categories = all_rels['rel_categories']
        rel_annotations = all_rels['train']
        categories = categories['categories']

        label_embeddings = []
        image_ids = []
        sql = """INSERT INTO "Image2GraphEmbedding" ("image_name","embeding_value","triplets") VALUES(%s,%s,%s);"""
        for triplets in tqdm(triplets_list):

            image_id = triplets['image_id']
            rel_target = rel_annotations[str(image_id).replace('.jpg', '')]

            annotation_ids = coco.getAnnIds(imgIds=[int(image_id.replace('.jpg', ''))])
            anno_coco = coco.loadAnns(annotation_ids)

            triplets_txt = []
            rel_labels = []
            for item in rel_target:
                rel_txt = rel_categories[item[2]]
                sub = categories[anno_coco[item[0]]['category_id'] - 1]['name']
                obj = categories[anno_coco[item[1]]['category_id'] - 1]['name']

                rel_labels.append(rel_txt)
                triplets_txt.append(sub + ' ' + rel_txt + ' ' + obj)

            label_ids, label_attention_mask = tokenize_triplets(triplets_txt)
            
            label_ids = label_ids.unsqueeze(0).to(device)
            label_attention_mask = label_attention_mask.unsqueeze(0).to(device)

            # label_ids = label_ids.to(device)
            # label_attention_mask = label_attention_mask.to(device)
            
            with torch.no_grad():
                
                # x, label_features = model.label_encoder(label_ids, label_attention_mask)
                label_features = model.label_encoder(label_ids, label_attention_mask)
            
            label_embeddings.append(label_features)
            image_ids.append(image_id)

            cursor.execute(sql,(image_id,label_features.tolist(),triplets_txt) )
            conn.commit()

            # break

        #print(len(image_ids))
        
        return torch.cat(label_embeddings, dim=0)
    except Exception as e:
        print(str(e))

    finally:
        # close connection
        if conn is not None:
            conn.close()
            #print("Database connection closed.")

@rev_api.route('/rev', methods = ['POST'])
@cross_origin()
def find_matcher():
    try: 
        data = request.get_json()
        text_triplets = data['triplet']
        top_k = 9

        model, _ = build_model(device)
        model.load_state_dict(torch.load(args.ckpt_rev, map_location=device))
        model.eval()

        # connect to DB
        conn = psycopg2.connect(args.conn_str)
        cursor = conn.cursor()

        label_embeddings = []
        file_image = []
        triplets = []
        
        sql = """SELECT image_name, embeding_value, triplets FROM "Image2GraphEmbedding";"""
        cursor.execute(sql)
        records = cursor.fetchall()
        for record in records[:]:
            file_image.append(record[0])
            label_embeddings.append(record[1])
            triplets.append(extract_subject_relation_object(parse_postgres_array(record[2])))

        label_embeddings = torch.tensor(label_embeddings).squeeze(1)
        text_triplets = [triplet.replace('/', ' ') for triplet in text_triplets]
        text_embedding = compute_text_embedding(text_triplets, model, device)

        #Normalization:
        text_embedding = F.normalize(text_embedding, p=2, dim=1)
        label_embeddings = F.normalize(label_embeddings, p=2, dim=1)
        
        #For Cosine Similarity
        # cosine_similarity = nn.CosineSimilarity(dim=-1)
        # similarities = cosine_similarity(text_embedding, label_embeddings)
        # # best_match_idx = torch.argmax(similarities).item()
        # top_k_indices = torch.topk(similarities.squeeze(), top_k).indices
        # top_k_image_names = [file_image[idx] for idx in top_k_indices]

        #For Dot product Similarity
        dot_similarity = torch.matmul(text_embedding, label_embeddings.t())
        _, results = torch.topk(dot_similarity, k=top_k, dim=1)
        results = results.cpu().numpy()
        top_k_image_names = [[file_image[idx] for idx in indices] for indices in results]
        top_k_triplets = [[triplets[idx] for idx in indices] for indices in results]
        trip_by_im = []
    
        for i, t in zip(top_k_image_names[0], top_k_triplets[0]):
            t_json = {
                "image_id": i,
                "trip": t
            }
            trip_by_im.append(t_json)
        
        res = {
            "imgs": top_k_image_names[0],
            "triplets": trip_by_im
        }
        return jsonify(
            Data = res,
            Status = True, 
            Msg = 'OK'
        )

    except Exception as e:
        return jsonify(
            Data = None,
            Status = False, 
            Msg = f'Error: {e}'
        )
    finally:
        # close connection
        if conn is not None:
            conn.close()
            #print("Database connection closed.")
    

@rev_api.route('/images/<filename>')
def serve_image(filename):
    return send_from_directory('Datasets/VisualGenome/VG_100K', filename)