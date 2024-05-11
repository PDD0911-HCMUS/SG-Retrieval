import psycopg2
from RelTRnBert import build_model
import torch.nn.functional as F
import os
import torch
import ConfigArgs as args
from transformers import BertTokenizer
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm




class ConvertGrayToRGB(object):
    def __call__(self, image):
        if image.mode != 'RGB':
            return image.convert('RGB')
        return image

transform = T.Compose([
    T.Resize(512),
    ConvertGrayToRGB(),  # Chuyển ảnh xám thành RGB
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

device = torch.device(args.device if torch.cuda.is_available() else "cpu")
dual_model_encoder, dot_loss = build_model(num_projection_layers=1,
                                        projection_dims=256,
                                        dropout_rate=0.1
                                        )
dual_model_encoder.reltr.load_state_dict(torch.load('ckpt/reltr_weights_63.pth', map_location=device))
dual_model_encoder.bert.load_state_dict(torch.load('ckpt/bert_weights_63.pth', map_location=device))
dual_model_encoder.reltr.eval()
dual_model_encoder.bert.eval()



def create_database():
    image_name = []
    image_embed = []
    try:
        # connect to DB
        conn = psycopg2.connect(args.conn_str)
        print("Connected to the database.")

        # create a cusor
        cursor = conn.cursor()

        print(type(conn))

        # exxcute query
        cursor.execute("SELECT version();")

        # get result
        record = cursor.fetchone()
        print("You are connected to - ", record)
        
        
        with torch.no_grad():
            sql = """INSERT INTO "Image2GraphEmbedding" ("image_name","embeding_value") VALUES(%s,%s);"""
            for item in tqdm(os.listdir(args.img_folder_vg)[:]):
                try:
                    # image_name.append(args.img_folder_vg + item)
                    image_name.append(item)
                    im = Image.open(args.img_folder_vg + item)
                    img = transform(im).unsqueeze(0)
                    sg_norm = dual_model_encoder.reltr(img)
                    em = sg_norm[0].tolist()
                    

                    cursor.execute(sql,(item,em,) )
                    conn.commit()

                    image_embed.append(sg_norm)
                except:
                    continue

        image_embed = torch.cat(image_embed, dim=0)

        print(f"Compelted creating database with {len(image_name)} number of images")
        print(f"Size of database: {image_embed.size()}")
        
    except Exception as e:
        print("Unable to connect to the database:", str(e))

    finally:
        # close connection
        if conn is not None:
            conn.close()
            print("Database connection closed.")

def preprocess_qu(text):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # txt = '[CLS] The Similar Function Loss will be based on Dot product similarity [SEP]'
    encoded_dict = tokenizer.encode_plus(
                            text,                  
                            add_special_tokens=True,   # Add '[CLS]' and '[SEP]'
                            max_length=256,             # Adjust sentence length
                            pad_to_max_length=True,    # Pad/truncate sentences
                            return_attention_mask=True,# Generate attention masks
                            return_tensors='pt',       # Return PyTorch tensors
                    )
    tensor_que = encoded_dict['input_ids']

    with torch.no_grad():
        txt_out = dual_model_encoder.bert(tensor_que)

    return txt_out

def find_matches(query, k=9, normalize=True):
    try:
        # connect to DB
        conn = psycopg2.connect(args.conn_str)
        print("Connected to the database.")
        # create a cusor
        cursor = conn.cursor()
        # exxcute query
        cursor.execute("SELECT version();")
        # get result
        record = cursor.fetchone()
        print("You are connected to - ", record)
        image_embeddings = []
        file_image = []
        
        sql = """SELECT image_name, embeding_value FROM "Image2GraphEmbedding";"""
        cursor.execute(sql)
        records = cursor.fetchall()
        for record in records[:]:
            file_image.append(record[0])
            image_embeddings.append(record[1])

        # Đóng cursor và kết nối
        cursor.close()
        conn.close()
        image_embeddings = torch.tensor(image_embeddings)
        print(image_embeddings.size())
        # print(image_embeddings)
        # image_embeddings = torch.cat(image_embeddings, dim=0)

    except Exception as e:
        print("Unable to connect to the database:", str(e))

    finally:
        # Đclose connection
        if conn is not None:
            conn.close()
            print("Database connection closed.")
    # Normalize the query and the image embeddings.
    query_embeddings = preprocess_qu(query)
    if normalize:
        image_embeddings = F.normalize(image_embeddings, p=2, dim=1)
        query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
    
    # Compute the dot product between the query and the image embeddings.
    dot_similarity = torch.mm(query_embeddings, image_embeddings.t())
    
    # Retrieve top k values and indices.
    values, indices = torch.topk(dot_similarity, k, dim=1)
    print(indices)
    # Return matching indices and their corresponding values.
    return indices, file_image


if __name__ == "__main__":
    txt = "The Similar Function Loss will be based on Dot product similarity"

    indices, _ = find_matches(txt, k=9, normalize=True)
    pass