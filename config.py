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
    # PASSWORD = "123456"
    PASSWORD = "09111997"
    PORT = 5432 #5432 for Local #5433 for csMachine
    SQLALCHEMY_DATABASE_URI = f"postgresql://{USERNAME}:{PASSWORD}@{HOSTNAME}:{PORT}/{DATABASE}"
    SQLALCHEMY_TRACK_MODIFICATIONS = False

class ConfigAPI:
    access_token_time = 15
    token_type = "Bearer"

class ConfigData:
    root = os.path.join(pwd,'Datasets')
    
    img_folder_vg = os.path.join(root,'VisualGenome/VG_100K/')
    img_folder_coco = os.path.join(root,'MSCOCO/mscoco')
    dir_upload =  os.path.join(root,'upload/')

    img_folder_mscoco_train = os.path.join(root,'MSCOCO/train2017/')
    img_folder_mscoco_valid = os.path.join(root,'MSCOCO/val2017/')

    #For HybridEncoderRegionDescriptionController
    hybrid_encoder_train = os.path.join(root, 'VisualGenome', 'anno_rg/train_data.json')
    hybrid_encoder_valid = os.path.join(root, 'VisualGenome', 'anno_rg/val_data.json')

    #For IRESGController
    # iresg_train = os.path.join(root, 'VisualGenome', 'anno_iresg/train.json')
    iresg_valid = os.path.join(root, 'VisualGenome', 'anno_iresg/valid.json')
    # iresg_test = os.path.join(root, 'VisualGenome', 'anno_iresg/test.json')
    # iresg_rel = os.path.join(root, 'VisualGenome', 'anno_reltr/rel.json')
    iresg_anno = os.path.join(root, 'VisualGenome', 'Rev_v2.json')

    iresg_train = os.path.join(root, 'MSCOCO', 'anno_iresg/train.json')
    # iresg_valid = os.path.join(root, 'MSCOCO', 'anno_iresg/valid.json')
    iresg_test = os.path.join(root, 'MSCOCO', 'anno_iresg/test.json')
    iresg_rel = os.path.join(root, 'MSCOCO', 'anno_reltr/rel.json')
    # iresg_anno = os.path.join(root, 'MSCOCO', 'Rev_v2_mscoco.json')

class ConfigDataBDD:
    root = os.path.join(pwd,'Datasets')
    image_folfer = os.path.join(root, 'BDD/bdd100k/bdd100k/images/100k')
    image_seg_folder = os.path.join(root, 'BDD/bdd100k_labels_release/bdd100k/labels/segment')
    anno_folder = os.path.join(root, 'BDD/bdd100k_labels_release/bdd100k/labels')

    train_driveable_anno = os.path.join(anno_folder, 'bdd100k_train_driveable.json')
    train_lane_anno = os.path.join(anno_folder, 'bdd100k_train_lane.json')
    train_box_anno = os.path.join(anno_folder, 'bdd100k_train_box2d.json')

    valid_driveable_anno = os.path.join(anno_folder, 'bdd100k_valid_driveable.json')
    valid_lane_anno = os.path.join(anno_folder, 'bdd100k_valid_lane.json')
    valid_box_anno = os.path.join(anno_folder, 'bdd100k_valid_box2d.json')

    categories = os.path.join(anno_folder, 'catgories.json')

class Checkpoint:
    root = os.path.join(pwd,'Checkpoint')
    ckpt_IRESGCL = os.path.join(root,'IRESGCL', 'model_epoch_80.pth')                                 
    ckpt_sgg = os.path.join(root,'RelTR','checkpoint0149reltr.pth')
    # ckpt_IRESG = os.path.join(root,'IRESG', 'epoch_39_mscoco.pth')
    ckpt_IRESG = os.path.join(root,'IRESG', 'epoch_39.pth')

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

