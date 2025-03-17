from flask_sqlalchemy import SQLAlchemy
db = SQLAlchemy()

root = '0_Datasets/Incidents/anno/'
anno_train = '0_Datasets/VisualGenome/train.json'
anno_valid = '0_Datasets/VisualGenome/val.json'
img_folder_vg = '/home/duypd/ThisPC-DuyPC/SG-Retrieval/0_Datasets/VisualGenome/VG_100K/'
dir_upload = '0_Datasets/upload/'

img_folder_mscoco = '/home/duypd/ThisPC-DuyPC/SG-Retrieval/0_Datasets/MSCOCO/train2017/'

batch_size = 12
num_workers = 0
num_epochs = 200
max_length = 128
device = 'cuda:2'
seed = 42

# connection information
hostname = 'localhost'
database = 'RetrievalSystemTraffic'
username = 'postgres'
password = '123456'
port_id = 5432
conn_str = f"dbname='{database}' user='{username}' host='{hostname}' password='{password}' port='{port_id}'"

class Config:
    HOSTNAME = "localhost"
    DATABASE = "RetrievalSystemTraffic"
    USERNAME = "postgres"
    PASSWORD = "123456"
    PORT = 5432

    SQLALCHEMY_DATABASE_URI = f"postgresql://{USERNAME}:{PASSWORD}@{HOSTNAME}:{PORT}/{DATABASE}"
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    CORS_HEADERS = "Content-Type"

CLASSES = [ 'N/A', 'airplane', 'animal', 'arm', 'bag', 'banana', 'basket', 'beach', 'bear', 'bed', 'bench', 'bike',
                'bird', 'board', 'boat', 'book', 'boot', 'bottle', 'bowl', 'box', 'boy', 'branch', 'building',
                'bus', 'cabinet', 'cap', 'car', 'cat', 'chair', 'child', 'clock', 'coat', 'counter', 'cow', 'cup',
                'curtain', 'desk', 'dog', 'door', 'drawer', 'ear', 'elephant', 'engine', 'eye', 'face', 'fence',
                'finger', 'flag', 'flower', 'food', 'fork', 'fruit', 'giraffe', 'girl', 'glass', 'glove', 'guy',
                'hair', 'hand', 'handle', 'hat', 'head', 'helmet', 'hill', 'horse', 'house', 'jacket', 'jean',
                'kid', 'kite', 'lady', 'lamp', 'laptop', 'leaf', 'leg', 'letter', 'light', 'logo', 'man', 'men',
                'motorcycle', 'mountain', 'mouth', 'neck', 'nose', 'number', 'orange', 'pant', 'paper', 'paw',
                'people', 'person', 'phone', 'pillow', 'pizza', 'plane', 'plant', 'plate', 'player', 'pole', 'post',
                'pot', 'racket', 'railing', 'rock', 'roof', 'room', 'screen', 'seat', 'sheep', 'shelf', 'shirt',
                'shoe', 'short', 'sidewalk', 'sign', 'sink', 'skateboard', 'ski', 'skier', 'sneaker', 'snow',
                'sock', 'stand', 'street', 'surfboard', 'table', 'tail', 'tie', 'tile', 'tire', 'toilet', 'towel',
                'tower', 'track', 'train', 'tree', 'truck', 'trunk', 'umbrella', 'vase', 'vegetable', 'vehicle',
                'wave', 'wheel', 'window', 'windshield', 'wing', 'wire', 'woman', 'zebra']

REL_CLASSES = ['__background__', 'above', 'across', 'against', 'along', 'and', 'at', 'attached to', 'behind',
                'belonging to', 'between', 'carrying', 'covered in', 'covering', 'eating', 'flying in', 'for',
                'from', 'growing on', 'hanging from', 'has', 'holding', 'in', 'in front of', 'laying on',
                'looking at', 'lying on', 'made of', 'mounted on', 'near', 'of', 'on', 'on back of', 'over',
                'painted on', 'parked on', 'part of', 'playing', 'riding', 'says', 'sitting on', 'standing on',
                'to', 'under', 'using', 'walking in', 'walking on', 'watching', 'wearing', 'wears', 'with']

prefix_name = 'object+'
prefix_graph = 'graph+'
prefix_triplet = 'triplet+'

ckpt_rev = 'ckpt/cross_modal_model_with_attention_epoch__30.pth'
ckpt_sgg = '/home/duypd/ThisPC-DuyPC/SG-Retrieval/ckpt/checkpoint0149reltr.pth'
ckpt_blip = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model*_base_caption.pth'