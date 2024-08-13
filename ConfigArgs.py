root = 'Datasets/Incidents/anno/'
# anno_train = root.replace('anno', 'train')
# anno_valid = root.replace('anno', 'val')
anno_train = 'Datasets/VisualGenome/train.json'
anno_valid = 'Datasets/VisualGenome/val.json'
img_folder_vg = 'Datasets/VisualGenome/VG_100K/'

batch_size = 12
num_workers = 0

num_epochs = 200

max_length = 128

device = 'cuda:2'

seed = 42