import json 
import os
from tqdm import tqdm

class DataCreation:
    def __init__(self, root, train_anno, valid_anno):
        self.root = root

        with open(os.path.join(self.root, train_anno), 'r') as f:
            self.train_anno = json.load(f)

        with open(os.path.join(self.root, valid_anno), 'r') as f:
            self.valid_anno = json.load(f)

    @staticmethod
    def _filter_poly2d_by_closed(anno, closed: bool):
        out = []
        for a in tqdm(anno):
            labels = a.get('labels') or []
            keep = [
                item for item in labels
                if any(p.get('closed') == closed for p in item.get('poly2d', []))
            ]
            out.append({
                'image_id': a.get('name'),
                'labels': keep
            })

        return out

    @staticmethod
    def get_poly2d_driveable(anno):
        return DataCreation._filter_poly2d_by_closed(anno, True)

    @staticmethod
    def get_poly2d_lane(anno):
        return DataCreation._filter_poly2d_by_closed(anno, False)

    @staticmethod
    def get_box2d(anno):
        box2d = []
        for a in tqdm(anno):
            labels = a.get('labels') or a.get('labels') or []
            keep = [item for item in labels if 'box2d' in item]
            box2d.append({
                'image_id': a.get('name'),
                'labels': keep
            })
        return box2d

    def run(self):

        train_driveable = self.get_poly2d_driveable(self.train_anno)
        valid_driveable = self.get_poly2d_driveable(self.valid_anno)

        train_lane = self.get_poly2d_lane(self.train_anno)
        valid_lane = self.get_poly2d_lane(self.valid_anno)

        train_box2d = self.get_box2d(self.train_anno)
        valid_box2d = self.get_box2d(self.valid_anno)

        with open(os.path.join(self.root, "bdd100k_train_driveable.json"), "w") as outfile:
            json.dump(train_driveable, outfile)

        with open(os.path.join(self.root, "bdd100k_valid_driveable.json"), "w") as outfile:
            json.dump(valid_driveable, outfile)

        with open(os.path.join(self.root, "bdd100k_train_lane.json"), "w") as outfile:
            json.dump(train_lane, outfile)

        with open(os.path.join(self.root, "bdd100k_valid_lane.json"), "w") as outfile:
            json.dump(valid_lane, outfile)

        with open(os.path.join(self.root, "bdd100k_train_box2d.json"), "w") as outfile:
            json.dump(train_box2d, outfile)

        with open(os.path.join(self.root, "bdd100k_valid_box2d.json"), "w") as outfile:
            json.dump(valid_box2d, outfile)

if __name__ == "__main__":
    root = "/home/duypd/ThisPC-DuyPC/SG-Retrieval/Datasets/BDD/bdd100k_labels_release/bdd100k/labels"
    train_anno = "bdd100k_labels_images_train.json"
    valid_anno = "bdd100k_labels_images_val.json"

    data_cre = DataCreation(root, train_anno, valid_anno)

    data_cre.run()

