root = 'Datasets/Incidents/anno/'
# anno_train = root.replace('anno', 'train')
# anno_valid = root.replace('anno', 'val')
anno_train = 'Datasets/VisualGenome/train.json'
anno_valid = 'Datasets/VisualGenome/val.json'
img_folder_vg = 'Datasets/VisualGenome/VG_100K/'

batch_size = 2
num_workers = 0

num_epochs = 100

max_length = 128

device = 'cuda:2'

seed = 42


# connection information
hostname = 'localhost'
database = 'RetrievalSystemTraffic'
username = 'postgres'
password = '123456'
port_id = 5432
# connection string
conn_str = f"dbname='{database}' user='{username}' host='{hostname}' password='{password}' port='{port_id}'"