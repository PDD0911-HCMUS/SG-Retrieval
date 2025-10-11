import json
import torch 
import os
from PIL import Image
from torch.utils.data import Dataset
class BDDDataset(Dataset):

    def __init__(self, image_folder, annotation):

        self.image_folder = image_folder
        self.annotation = annotation 
        


