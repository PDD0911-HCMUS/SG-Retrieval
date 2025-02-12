import json
from torch.utils.data import Dataset

class CreateData(Dataset):
    def __init__(self, img_folder, ann_file, transforms, return_masks):
        with open(ann_file, 'r') as f:
            self.data = json.load(f)

        self._transforms = transforms

    def get_item(self, idx):
        pass