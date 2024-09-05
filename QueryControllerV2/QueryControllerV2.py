from QueryController.datasets.data_pre import build_data, tokenize_triplets
from QueryController.model_cross import build_model
from pycocotools.coco import COCO
import torch.nn.functional as F
import torch
import torch.nn as nn
from tqdm import tqdm
import json
import psycopg2
# import ConfigArgs as args

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# connection information
hostname = 'localhost'
database = 'RetrievalSystemTraffic'
username = 'postgres'
password = '123456'
port_id = 5432
# connection string
conn_str = f"dbname='{database}' user='{username}' host='{hostname}' password='{password}' port='{port_id}'"

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
        conn = psycopg2.connect(conn_str)
        print("Connected to the database.")

        # create a cusor
        cursor = conn.cursor()

        print(type(conn))

        # exxcute query
        cursor.execute("SELECT version();")

        # get result
        record = cursor.fetchone()
        print("You are connected to - ", record)

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

        print(len(image_ids))
        
        return torch.cat(label_embeddings, dim=0)
    except Exception as e:
        print("Unable to connect to the database:", str(e))

    finally:
        # close connection
        if conn is not None:
            conn.close()
            print("Database connection closed.")

def find_matcher(text_triplets, top_k = 10):
    try: 

        model, criterion = build_model(device)

        model.load_state_dict(torch.load('ckpt/cross_modal_model_with_attention_epoch__30.pth', map_location=device))

        model.eval()
        # connect to DB
        conn = psycopg2.connect(conn_str)
        print("Connected to the database.")

        # create a cusor
        cursor = conn.cursor()

        print(type(conn))

        # exxcute query
        cursor.execute("SELECT version();")

        # get result
        record = cursor.fetchone()
        print("You are connected to - ", record)

        label_embeddings = []
        file_image = []
        
        sql = """SELECT image_name, embeding_value FROM "Image2GraphEmbedding";"""
        cursor.execute(sql)
        records = cursor.fetchall()
        for record in records[:]:
            file_image.append(record[0])
            label_embeddings.append(record[1])

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

        return {
            "imgs": top_k_image_names[0]
        }
    except Exception as e:
        print(e)
    finally:
        # close connection
        if conn is not None:
            conn.close()
            print("Database connection closed.")

# def main():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")

#     model, criterion = build_model(device)

#     model.load_state_dict(torch.load('ckpt/cross_modal_model_with_attention_epoch__30.pth', map_location=device))

#     model.eval()

#     with open('Datasets/VisualGenome/train_trip.json', 'r') as f:
#         triplets_data = json.load(f)

#     text_triplets = ["man holding umbrella", "woman wearing hat", "dog running"]

#     # compute_label_embeddings(triplets_data, model, device)

#     # text_features = compute_text_embedding(text_triplets, model, device)
#     # print(text_features)

#     best_match_idx, im = query(text_triplets, model, device, top_k=5)
#     print(best_match_idx)
#     print(im)

# if __name__ == main():
#     main()
