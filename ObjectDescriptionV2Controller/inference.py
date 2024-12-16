import os 
from transformers import BertTokenizer
from .datasets.dataV3Posgres import build, custom_collate_fn
from torch.utils.data import DataLoader
from .model.Model import build_model
import numpy as np
import torch
from rich.progress import Progress
import psycopg2
import faiss
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sentence_transformers import SentenceTransformer
import json

hostname = 'localhost'
database = 'RetrievalSystemTraffic'
username = 'postgres'
password = '123456'
port_id = 5432
conn_str = f"dbname='{database}' user='{username}' host='{hostname}' password='{password}' port='{port_id}'"

def parse_postgres_array(data):
    cleaned_data = data.replace('{', '').replace('}', '').replace('""', '"')
    list_data = cleaned_data.split('","')
    list_data = [item.strip('"') for item in list_data]
    
    return list_data

def pad_or_truncate_tensor(item, max_i = 10):
    # Lấy kích thước hiện tại của tensor
    for i in item.keys():
        if(i == 'trip' or i == 'trip_msk'):
            if(item[i].size(0) < max_i):
                num_i, _ = item[i].size()
                expand = torch.zeros((max_i - num_i, max_i), dtype=item[i].dtype)
                item[i] = torch.cat([item[i], expand], dim = 0)
            if(item[i].size(0) > max_i):
                item[i] = item[i][: max_i]
    return item

def create_input_data(triplet: list):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    maxlenght = 10
    encoded_triplets_que = []
    labels_que_s = []
    labels_que_o = []
    labels_que_r = []

    item_q = {}

    for item_que in triplet:
            words = item_que.split(' ')
            encoded_triplets_que.append(item_que)
            labels_que_s.append(words[0])
            labels_que_o.append(words[-1])
            labels_que_r.append(' '.join(words[1:-1]))

    labels_que_s = [' '.join(labels_que_s)]
    labels_que_o = [' '.join(labels_que_o)]
    labels_que_r = [' '.join(labels_que_r)]

    e_s_que = tokenizer(labels_que_s, padding = 'max_length', truncation = True, max_length = maxlenght, return_tensors = 'pt')
    e_o_que = tokenizer(labels_que_o, padding = 'max_length', truncation = True, max_length = maxlenght, return_tensors = 'pt')
    e_r_que = tokenizer(labels_que_r, padding = 'max_length', truncation = True, max_length = maxlenght, return_tensors = 'pt')
    e_t_que = tokenizer(encoded_triplets_que, padding = 'max_length', truncation = True, max_length = maxlenght, return_tensors = 'pt')

    item_q['sub_labels'] = e_s_que['input_ids']
    item_q['obj_labels'] = e_o_que['input_ids']
    item_q['rel_labels'] = e_r_que['input_ids']
    item_q['trip'] = e_t_que['input_ids']
    item_q['sub_labels_msk'] = e_s_que['attention_mask']
    item_q['obj_labels_msk'] = e_o_que['attention_mask']
    item_q['rel_labels_msk'] = e_r_que['attention_mask']
    item_q['trip_msk'] = e_t_que['attention_mask']

    item = pad_or_truncate_tensor(item_q)
    return [item]

def create_input_data_all_MiniLM_L6_v2(triplet: list):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    maxlenght = 10
    encoded_triplets_que = []
    labels_que_s = []
    labels_que_o = []
    labels_que_r = []

    item_q = {}

    for item_que in triplet:
            words = item_que.split(' ')
            encoded_triplets_que.append(item_que)
            labels_que_s.append(words[0])
            labels_que_o.append(words[-1])
            labels_que_r.append(' '.join(words[1:-1]))

    labels_que_s = [' '.join(labels_que_s)]
    labels_que_o = [' '.join(labels_que_o)]
    labels_que_r = [' '.join(labels_que_r)]

    e_s_que = tokenizer(labels_que_s, padding = 'max_length', truncation = True, max_length = maxlenght, return_tensors = 'pt')
    e_o_que = tokenizer(labels_que_o, padding = 'max_length', truncation = True, max_length = maxlenght, return_tensors = 'pt')
    e_r_que = tokenizer(labels_que_r, padding = 'max_length', truncation = True, max_length = maxlenght, return_tensors = 'pt')
    e_t_que = tokenizer(encoded_triplets_que, padding = 'max_length', truncation = True, max_length = maxlenght, return_tensors = 'pt')

    item_q['sub_labels'] = e_s_que['input_ids']
    item_q['obj_labels'] = e_o_que['input_ids']
    item_q['rel_labels'] = e_r_que['input_ids']
    item_q['trip'] = e_t_que['input_ids']
    item_q['sub_labels_msk'] = e_s_que['attention_mask']
    item_q['obj_labels_msk'] = e_o_que['attention_mask']
    item_q['rel_labels_msk'] = e_r_que['attention_mask']
    item_q['trip_msk'] = e_t_que['attention_mask']

    item = pad_or_truncate_tensor(item_q)
    return [item]

def get_input_embedding(input, model: torch.nn.Module):
    item = create_input_data(input)
    model.eval()
    with torch.no_grad():
        x, _ = model(item,item)
    return x

def make_dataset_posgres_insert(out_que, out_rev, que_im, rev_im):
    # connect to DB
    try:
        conn = psycopg2.connect(conn_str)
        cursor = conn.cursor()
        sql = """INSERT INTO "GraphRetrieval_V2" ("image_name_qu","que_embeding","image_name_rev","rev_embeding") VALUES(%s,%s,%s,%s);"""
        for o_q, o_r, q_i, r_i in zip(out_que, out_rev, que_im, rev_im):
            # print(q_i, o_q.size(), r_i, o_r.size())
            cursor.execute(sql,(q_i,o_q.tolist(),r_i, o_r.tolist()) )
            conn.commit()
    except Exception as e:
        print(str(e))
    finally:
        if conn is not None:
            conn.close()

def make_dataset_posgres_insert_all_MiniLM_L6_v2(out_que, out_rev, que_im, rev_im):
    # connect to DB
    try:
        conn = psycopg2.connect(conn_str)
        cursor = conn.cursor()
        sql = """INSERT INTO "GraphRetrieval_all-MiniLM-L6-v2" ("image_name_qu","que_embeding","image_name_rev","rev_embeding") VALUES(%s,%s,%s,%s);"""
        cursor.execute(sql,(que_im,out_que.tolist(),rev_im, out_rev.tolist()) )
        conn.commit()
    except Exception as e:
        print(str(e))
    finally:
        if conn is not None:
            conn.close()

def make_dataset_posgres(model: torch.nn.Module, device: torch.device):

    model.eval()
    root_data = '/radish/phamd/duypd-proj/SG-Retrieval/Datasets/VisualGenome/'
    ann_file = root_data + 'Rev_v2.json'
    dataset = build(ann_file)
    dataloader = DataLoader(dataset, batch_size=32, collate_fn=custom_collate_fn)

    with Progress() as progress:
            task = progress.add_task("[cyan]Creating...", total=len(dataloader))
            for batch_idx, (que, rev) in enumerate(dataloader):
                que_i = [{k: v.to(device) for k, v in t.items() if k != 'image_id'} for t in que]
                que_im = [{k: v for k, v in t.items() if k == 'image_id'} for t in que ]

                rev_i = [{k: v.to(device) for k, v in t.items() if k != 'image_id'} for t in rev]
                rev_im = [{k: v for k, v in t.items() if k == 'image_id'} for t in rev ]

                out_que, out_rev = model(que_i, rev_i)
                q_im = [item['image_id'] for item in que_im]
                r_im = [item['image_id'] for item in rev_im]

                make_dataset_posgres_insert(out_que, out_rev, q_im, r_im)

                progress.update(task, advance=1, description=f"[cyan]Batch {batch_idx+1}/{len(dataloader)}")

def make_dataset_posgres_all_MiniLM_L6_v2(model: torch.nn.Module, device: torch.device):

    model.eval()
    root_data = '/radish/phamd/duypd-proj/SG-Retrieval/Datasets/VisualGenome/'
    ann_file = root_data + 'Rev_v2.json'
    # dataset = build_v2(ann_file)
    # dataloader = DataLoader(dataset, batch_size=1)

    with open(ann_file, 'r') as f:
        data = json.load(f)

    with Progress() as progress:
            task = progress.add_task("[cyan]Creating...", total=len(data))
            for batch_idx, item in enumerate(data):

                encoded_triplets_que = [", ".join(item['qe']['trip'])]
                encoded_triplets_rev = [", ".join(item['rev']['trip'])]

                # out_que, out_rev = model(que_i, rev_i)
                out_que = model.encode(encoded_triplets_que)
                out_que = torch.tensor(out_que)

                out_rev = model.encode(encoded_triplets_rev)
                out_rev = torch.tensor(out_rev)

                q_im = item['qe']["image_id"]
                r_im = item['rev']["image_id"]

                make_dataset_posgres_insert_all_MiniLM_L6_v2(out_que[0], out_rev[0], q_im, r_im)

                progress.update(task, advance=1, description=f"[cyan]Batch {batch_idx+1}/{len(data)}")
                # break

def get_set_query():
    conn = psycopg2.connect(conn_str)
    cursor = conn.cursor()
    sql = """SELECT image_name_qu, image_name_rev, que_embeding, rev_embeding FROM "GraphRetrieval_V2";"""
    cursor.execute(sql)
    records = cursor.fetchall()

    file_image_q = []
    file_image_r = []
    que_embeding = []
    rev_embeding = []
    for record in records[:]:
        file_image_q.append(record[0])
        file_image_r.append(record[1])
        que_embeding.append(record[2])
        rev_embeding.append(record[3])

    images = file_image_q + file_image_r
    set_embedding = que_embeding + rev_embeding
    set_embedding = np.array(set_embedding)
    images, unique_indices = np.unique(images, return_index=True)
    set_embedding = set_embedding[unique_indices]
    return images, set_embedding

def get_set_query_all_MiniLM_L6_v2():
    conn = psycopg2.connect(conn_str)
    cursor = conn.cursor()
    sql = """SELECT image_name_qu, image_name_rev, que_embeding, rev_embeding FROM "GraphRetrieval_all-MiniLM-L6-v2";"""
    cursor.execute(sql)
    records = cursor.fetchall()

    file_image_q = []
    file_image_r = []
    que_embeding = []
    rev_embeding = []
    for record in records[:]:
        file_image_q.append(record[0])
        file_image_r.append(record[1])
        que_embeding.append(record[2])
        rev_embeding.append(record[3])

    images = file_image_q + file_image_r
    set_embedding = que_embeding + rev_embeding
    set_embedding = np.array(set_embedding)

    images, unique_indices = np.unique(images, return_index=True)
    set_embedding = set_embedding[unique_indices]

    return images, set_embedding

def faiss_retrieval_controller(trip, model):
    images, set_embedding = get_set_query()
    input_embedding = get_input_embedding(trip, model)
    input_embedding = input_embedding.numpy().astype('float32')
    index = faiss.IndexFlatIP(set_embedding.shape[1])  # Dùng Euclidean distance
    index.add(set_embedding)
    D, I = index.search(input_embedding, k=50)
    print("Indices:", I[0])
    selected_images = [images[i] for i in I[0]]
    return selected_images

def faiss_retrieval_controller_all_MiniLM_L6_v2(trip, model):
    images, set_embedding = get_set_query_all_MiniLM_L6_v2()

    print(set_embedding.shape)
    input_embedding = [", ".join(trip)]
    input_embedding = model.encode(input_embedding)
    input_embedding = torch.tensor(input_embedding)
    input_embedding = input_embedding.numpy().astype('float32')

    print(input_embedding.shape)

    index = faiss.IndexFlatIP(set_embedding.shape[1])  # Dùng Euclidean distance
    index.add(set_embedding)
    D, I = index.search(input_embedding, k=50)
    print("Indices:", I[0])
    selected_images = [images[i] for i in I[0]]
    return selected_images

def get_model():
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    return model

def calculate_precision_at_k(ground_truth, retrieved_results, k):
    top_k_results = retrieved_results[:k]
    relevant_in_top_k = len(set(top_k_results) & set(ground_truth))
    precision = relevant_in_top_k / k
    return precision

def jaccard_similarity(set1, set2):
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    if not union:
        return 0.0  # Tránh chia cho 0
    return len(intersection) / len(union)

def get_set():
    imgs , queries = get_set_query()
    query = """
    SELECT image_name, triplets
    FROM "Image2GraphEmbedding_V2"
    WHERE image_name = ANY(%s);
    """
    conn = psycopg2.connect(conn_str)
    cursor = conn.cursor()
    try:
        with conn.cursor() as cursor:
            cursor.execute(query, (imgs.tolist(),))
            results = cursor.fetchall()
    finally:
        conn.close()
    image_set = []
    triplet_set = []
    for row in results:
        # print(f"Image Name: {row[0]}, Triplets: {row[1]}")
        image_set.append(row[0])
        triplet_set.append(set(parse_postgres_array(row[1])))

    return image_set, triplet_set
def main():

    # root_ckpt = '/home/duypd/ThisPC-DuyPC/SG-Retrieval/ObjectDescriptionV2Controller/ckpt/'
    # ckpt = root_ckpt + 'model_epoch_80.pth'
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model, _ = build_model(d_model=256, dropout=0.1, activation="relu", pretrain = 'bert-base-uncased', device = device)
    # model.load_state_dict(torch.load(ckpt, map_location=device))

    model = get_model()
    
    # make_dataset_posgres_all_MiniLM_L6_v2(model, device)
    # make_dataset_posgres(model,device)

    trip =  [
        "fork on table",
                "plate on table",
                "fork near plate",
                "fork on plate",
                "food on plate"
    ]

    selected_images = faiss_retrieval_controller_all_MiniLM_L6_v2(trip, model)
    print(selected_images)
    print('Done')
    image_set, triplet_set = get_set()
    text_triplets = set(trip)
    top_k = 100
    combined = list(zip(triplet_set, image_set))
    similarities = [(filename, jaccard_similarity(text_triplets, triplet_set), triplet_set) for triplet_set, filename in combined]
    top_k_res = sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]

    img_id = []
    trip_by_im = []
    for i, s, t in top_k_res:
        t_json = {
            "image_id": i,
            "trip": t
        }
        img_id.append(i)
        trip_by_im.append(t_json)

    P_5 = calculate_precision_at_k(img_id, selected_images, k=10)
    P_10 = calculate_precision_at_k(img_id, selected_images, k=50)
    P_50 = calculate_precision_at_k(img_id, selected_images, k=100)

    print(P_5, P_10, P_50)

    # plt.figure(figsize=(15, 10)) 
    # for idx, file in enumerate(selected_images):  
    #     img = mpimg.imread('/home/duypd/ThisPC-DuyPC/SG-Retrieval/Datasets/VisualGenome/VG_100K/' + file)  
    #     plt.subplot(10, 5, idx + 1)  
    #     plt.imshow(img)
    #     plt.axis('off') 
    #     plt.title(f"Image {idx + 1}") 

    # plt.tight_layout()
    # plt.show()


if __name__ == "__main__":

    main()
