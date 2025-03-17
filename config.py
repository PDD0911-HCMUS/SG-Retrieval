from flask_sqlalchemy import SQLAlchemy
import os

pwd = os.getcwd()
db = SQLAlchemy()

class ConfigApp:
    domain = "localhost"
    port = 8009
    CORS_HEADER = 'Content-Type'
    

class ConfigDB:
    HOSTNAME = "localhost"
    DATABASE = "RetrievalSystemTraffic"
    USERNAME = "postgres"
    PASSWORD = "123456"
    PORT = 5432
    SQLALCHEMY_DATABASE_URI = f"postgresql://{USERNAME}:{PASSWORD}@{HOSTNAME}:{PORT}/{DATABASE}"
    SQLALCHEMY_TRACK_MODIFICATIONS = False

class ConfigData:
    root = os.path.join(pwd,'Datasets')
    
    img_folder_vg = os.path.join(root,'VisualGenome/VG_100K/')
    dir_upload =  os.path.join(root,'upload/')

    img_folder_mscoco_train = os.path.join(root,'MSCOCO/train2017/')
    img_folder_mscoco_valid = os.path.join(root,'MSCOCO/val2017/')

    #For CrossEncoderController

    cross_encoder_train = os.path.join(root, 'VisualGenome', 'anno_rg/train_data.json')
    cross_encoder_valid = os.path.join(root, 'VisualGenome', 'anno_rg/val_data.json')


    anno_train = '0_Datasets/VisualGenome/train.json'
    anno_valid = '0_Datasets/VisualGenome/val.json'

class Checkpoint:
    root = os.path.join(pwd,'Checkpoint')
    ckpt_IRESGCL = os.path.join(root,'IRESGCL', 'model_epoch_80.pth')
    ckpt_sgg = os.path.join(root,'RelTR','checkpoint0149reltr.pth')

batch_size = 12
num_workers = 0
num_epochs = 200
max_length = 128
device = 'cuda:2'
seed = 42

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

