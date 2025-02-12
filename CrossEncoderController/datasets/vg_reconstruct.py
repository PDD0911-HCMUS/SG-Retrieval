import os
import json
import random
from PIL import Image,ImageDraw
from scipy.spatial import distance
from rich.progress import Progress



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
        
def nearest_phase_bbox(bbox, region_mappings):
    x,y,w,h = bbox
    c_x = x + w/2
    c_y = y + h/2
    bbox_c = (c_x, c_y)

    nearest_phase = min(region_mappings, key=lambda r:distance.euclidean(r['center'], bbox_c))
    return nearest_phase

def create_region_mappings(region):
    region_mappings = []
    for region in region:
        x, y, w, h = region["bbox"]
        center_x = x + w / 2
        center_y = y + h / 2
        region_mappings.append({"center": (center_x, center_y), "phrase": region["phrase"], "bbox": region["bbox"]})

    return region_mappings

def main_re_construct():
    with open('/home/duypd/ThisPC-DuyPC/SG-Retrieval/Datasets/VisualGenome/anno_org/scene_graphs.json') as f:
        sg_json = json.load(f)

    with open('/home/duypd/ThisPC-DuyPC/SG-Retrieval/Datasets/VisualGenome/anno_org/region_descriptions.json') as f:
        rg_json = json.load(f)

    with open('/home/duypd/ThisPC-DuyPC/SG-Retrieval/Datasets/VisualGenome/anno_org/image_data.json') as f:
        image_json = json.load(f)

    images = []
    annotations = []
    id_counter = 1
    mode_train = True
    mode_valid = True
    mode_test = True
    combined = list(zip(sg_json, rg_json))  
    random.shuffle(combined)  # Shuffle cả danh sách
    sg_json, rg_json = zip(*combined)  # Tách lại thành hai danh sách

    # Nếu cần danh sách thay vì tuple
    sg_json = list(sg_json)
    rg_json = list(rg_json)

    with Progress() as progress:
        task = progress.add_task("[cyan]Creating VG annotation...", total=len(sg_json))
        for sg, rg in zip(sg_json, rg_json):

            if(sg['image_id'] == rg['id']):

                #Create images based on MSCOCO structure
                images.append(get_image_data(sg['image_id'], image_json))

                #Re-construct SG and RG
                return_objs_sg = re_construct_sg(sg)
                return_objs_rg = re_construct_rg(rg)
                region_mappings = create_region_mappings(return_objs_rg)

                #Create annotation based on MSCOCO structure
                for item in return_objs_sg:

                    phase_sub = nearest_phase_bbox(item['bbox_sub'], region_mappings)
                    phase_obj = nearest_phase_bbox(item['bbox_obj'], region_mappings)

                    # if(phase_sub == phase_obj):
                    #     continue

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
            
                if(mode_train == True and len(images) == 86461): #80% train
                    print("dumping for training set")
                    vg_anno = {
                        "images": images,
                        "annotations": annotations
                    }    

                    with open(os.getcwd() + '/vg_train.json', 'w') as f:
                        json.dump(vg_anno, f)

                    mode_train = False
                    images = []
                    annotations = []
                    print("Created training")
                if(mode_train == False and mode_valid == True and len(images) == 10807):#10% validation
                    print("dumping for validation set")
                    vg_anno = {
                        "images": images,
                        "annotations": annotations
                    }    

                    with open(os.getcwd() + '/vg_val.json', 'w') as f:
                        json.dump(vg_anno, f)
                    
                    mode_valid = False
                    images = []
                    annotations = []
                    
                    print("Created validation")

                if(mode_train == False and mode_valid == False and len(images) == 10809):#10% testing
                    print("dumping for testing set")
                    vg_anno = {
                        "images": images,
                        "annotations": annotations
                    }    

                    with open(os.getcwd() + '/vg_test.json', 'w') as f:
                        json.dump(vg_anno, f)
                    
                    print("Created testing")


    print("===================---Done---===================")

            # break

    # vg_anno = {
    #     "images": images,
    #     "annotations": annotations
    # }    

    # with open(os.getcwd() + '/vg.json', 'w') as f:
    #     json.dump(vg_anno, f)

    # print("Created")
    # pass

main_re_construct()