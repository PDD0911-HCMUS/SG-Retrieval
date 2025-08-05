import json 
import random

anno_file = '/home/duypd/ThisPC-DuyPC/SG-Retrieval/Datasets/MSCOCO/Rev_v2_mscoco.json'

with open(anno_file, 'r') as f:
    data = json.load(f)

print(len(data))

total = len(data)
train_split = int(0.9 * total)
valid_split = int(0.05 * total)

train_data = data[:train_split]
valid_data = data[train_split:train_split + valid_split]
test_data = data[train_split + valid_split:]

with open("/home/duypd/ThisPC-DuyPC/SG-Retrieval/Datasets/MSCOCO/anno_iresg/train.json", "w") as f:
    json.dump(train_data, f, indent=2)

with open("/home/duypd/ThisPC-DuyPC/SG-Retrieval/Datasets/MSCOCO/anno_iresg/valid.json", "w") as f:
    json.dump(valid_data, f, indent=2)

with open("/home/duypd/ThisPC-DuyPC/SG-Retrieval/Datasets/MSCOCO/anno_iresg/test.json", "w") as f:
    json.dump(test_data, f, indent=2)