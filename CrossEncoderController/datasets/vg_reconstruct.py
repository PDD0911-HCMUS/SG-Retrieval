import os
import json
import random
from PIL import Image,ImageDraw
from scipy.spatial import distance
from rich.progress import Progress

def compute_iou(bbox1, bbox2):
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    # Tọa độ của phần giao nhau
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)

    # Nếu không có giao nhau, IoU = 0
    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # Tính diện tích giao nhau
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Tính diện tích của từng bbox
    bbox1_area = w1 * h1
    bbox2_area = w2 * h2

    # Tính IoU
    iou = intersection_area / float(bbox1_area + bbox2_area - intersection_area)
    return iou

def re_construct_sg(sg):
    objs = sg['objects']
    return_objs = []
    object_dict = {obj["object_id"]: obj for obj in objs}
    for item in sg['relationships']:
        return_obj = {}
        found_obj_sub = object_dict.get(item['subject_id'])
        found_obj_obj = object_dict.get(item['object_id'])
        return_obj["bbox_sub"] = [found_obj_sub['x'], found_obj_sub['y'], found_obj_sub['w'], found_obj_sub['h']]
        return_obj["bbox_obj"] = [found_obj_obj['x'], found_obj_obj['y'], found_obj_obj['w'], found_obj_obj['h']]
        return_obj["predicate"] = item["predicate"]
        return_obj["relationship_id"] = item["relationship_id"]
        return_objs.append(return_obj)
    return return_objs

def re_construct_rg(rg):
    return_objs = []
    for item in rg['regions']:
        return_obj = {}
        return_obj["bbox"] = [item['x'], item['y'], item['width'], item['height']]
        return_obj["phrase"] = item["phrase"]
        return_obj["region_id"] = item["region_id"]
        return_objs.append(return_obj)
    return return_objs

def get_image_data(image_id, image_data):
    for x in image_data:
        if(x['image_id'] == image_id):
            item = {
                'width': x['width'],
                'height': x['height'],
                'id': x['image_id'],
                'file_name': str(x['image_id'])+'.jpg'
            }
            return item 
        
def nearest_phase_bbox(bbox, region_mappings, iou_threshold=0.7):

    best_match = None
    best_iou = 0
    best_distance = float("inf")

    x,y,w,h = bbox
    c_x = x + w/2
    c_y = y + h/2
    bbox_c = (c_x, c_y)

    for region in region_mappings:
        region_bbox = region["bbox"]
        region_center = (region_bbox[0] + region_bbox[2] / 2, region_bbox[1] + region_bbox[3] / 2)

        # Tính IoU
        iou = compute_iou(bbox, region_bbox)

        # Nếu IoU vượt ngưỡng, chọn ngay
        if iou > iou_threshold:
            return region

        # Nếu IoU thấp, fallback về Khoảng Cách Euclidean
        dist = distance.euclidean(bbox_c, region_center)
        if dist < best_distance:
            best_distance = dist
            best_match = region

    return best_match if best_match else {"phrase": "Unknown"} 

def create_region_mappings(region):
    region_mappings = []
    for region in region:
        x, y, w, h = region["bbox"]
        center_x = x + w / 2
        center_y = y + h / 2
        region_mappings.append({"center": (center_x, center_y), "phrase": region["phrase"], "bbox": region["bbox"]})

    return region_mappings

def prepare_data(sg_set, rg_set, id_counter, image_json, mode = 'train'):
    images = []
    annotations = []
    id_counter = id_counter
    with Progress() as progress:
        task = progress.add_task(f"[cyan]Creating VG {mode} annotation...", total=len(sg_set))
        for sg, rg in zip(sg_set, rg_set):
            if(sg['image_id'] == rg['id']):
                #Re-construct SG and RG
                return_objs_sg = re_construct_sg(sg)
                return_objs_rg = re_construct_rg(rg)
                region_mappings = create_region_mappings(return_objs_rg)

                #Create annotation based on MSCOCO structure
                # print(sg['image_id'],rg['id'])
                if(len(return_objs_sg) != 0 and len(return_objs_rg) != 0 ):
                    images.append(get_image_data(sg['image_id'], image_json))
                    for item in return_objs_sg:
                        phase_sub = nearest_phase_bbox(item['bbox_sub'], region_mappings)
                        phase_obj = nearest_phase_bbox(item['bbox_obj'], region_mappings)

                        anno = {}
                        anno["image_id"] = sg['image_id']
                        anno["iscrowd"] = 0
                        anno["id"] = id_counter
                        anno["category_id"] = 1
                        anno["segmentation"] = None
                        anno["area_sub"] = item['bbox_sub'][2] * item['bbox_sub'][3]
                        anno["area_obj"] = item['bbox_obj'][2] * item['bbox_obj'][3]
                        anno["bbox_sub"] = item['bbox_sub']
                        anno["bbox_obj"] = item['bbox_obj']
                        anno["phrase_sub"] = phase_sub['phrase']
                        anno["phrase_obj"] = phase_obj['phrase']
                        anno["predicate"] = item['predicate']
                        anno['predicate_id'] = item['relationship_id']

                        annotations.append(anno) 
                        id_counter += 1
                    progress.update(task, advance=1, description=f"[cyan] creating")

    vg_anno = {
        "images": images,
        "annotations": annotations
    }    

    with open(os.getcwd() + f"/vg_{mode}.json", 'w') as f:
        json.dump(vg_anno, f)

    print(f"Created vg_{mode}.json")
    return id_counter

def main_re_construct():
    with open('/home/duypd/ThisPC-DuyPC/SG-Retrieval/Datasets/VisualGenome/anno_org/scene_graphs.json') as f:
        sg_json = json.load(f)

    with open('/home/duypd/ThisPC-DuyPC/SG-Retrieval/Datasets/VisualGenome/anno_org/region_descriptions.json') as f:
        rg_json = json.load(f)

    with open('/home/duypd/ThisPC-DuyPC/SG-Retrieval/Datasets/VisualGenome/anno_org/image_data.json') as f:
        image_json = json.load(f)

    combined = list(zip(sg_json, rg_json))  
    random.shuffle(combined)  # Shuffle 
    sg_json, rg_json = zip(*combined) #tuple

    #convert to list
    sg_json = list(sg_json)
    rg_json = list(rg_json)

    #split train, val test: 80,10,10
    total_sapmles = len(sg_json)
    train_len = int(total_sapmles * 0.8)
    val_len = train_len + int(total_sapmles * 0.1)

    sg_train, rg_train = sg_json[:train_len], rg_json[:train_len]
    sg_val, rg_val = sg_json[train_len:val_len], rg_json[train_len:val_len]
    sg_test, rg_test = sg_json[val_len:], rg_json[val_len:]

    id_counter = 1 #defaut for the first time 

    id_counter_train = prepare_data(sg_train, rg_train, id_counter, image_json, 'train')
    id_counter_val = prepare_data(sg_val, rg_val, id_counter_train, image_json, 'val')
    _ = prepare_data(sg_test, rg_test, id_counter_val, image_json, 'test')


if __name__ == "__main__":
    main_re_construct()