#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BDD100K -> COCO (INTERSECTION: same image set for detection & driveable) + masks

- INPUT JSONs:
  Datasets/BDD/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_train.json
  Datasets/BDD/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_val.json

- IMAGE ROOT (given by you):
  /home/duypd/ThisPC-DuyPC/SG-Retrieval/Datasets/BDD/bdd100k/bdd100k/images/100k/{train,val,test}

- COCO OUTPUT (in labels/):
  instances_train2017.json, instances_val2017.json   (bbox-only)
  driveable_train2017.json, driveable_val2017.json   (polygons)

- MASK OUTPUT (in labels/segment/):
  labels/segment/train/*.png
  labels/segment/valid/*.png

Rule:
- Keep an image ONLY if it has (A) at least one closed polygon AND (B) at least one valid bbox.
- Thus instances_* and driveable_* have IDENTICAL image sets and ids.
"""

import os
import json
from tqdm import tqdm
from PIL import Image
import numpy as np
import cv2

# ---------- FIXED PATHS ----------
ROOT_LABELS = os.path.join("data", "bdd100k", "bdd100k_labels")
TRAIN_JSON = os.path.join(ROOT_LABELS, "bdd100k_labels_images_train.json")
VAL_JSON   = os.path.join(ROOT_LABELS, "bdd100k_labels_images_val.json")

IMAGE_ROOT_100K = "data/bdd100k/bdd100k_images_100k"

OUT_INST_TRAIN = os.path.join(ROOT_LABELS, "instances_train2017.json")
OUT_INST_VAL   = os.path.join(ROOT_LABELS, "instances_val2017.json")
OUT_DRV_TRAIN  = os.path.join(ROOT_LABELS, "driveable_train2017.json")
OUT_DRV_VAL    = os.path.join(ROOT_LABELS, "driveable_val2017.json")

MASK_DIR_TRAIN = os.path.join(ROOT_LABELS, "segment", "train")
MASK_DIR_VALID = os.path.join(ROOT_LABELS, "segment", "valid")
os.makedirs(MASK_DIR_TRAIN, exist_ok=True)
os.makedirs(MASK_DIR_VALID, exist_ok=True)

# ---------- CATEGORIES ----------
DET_CATEGORIES = [
    {"supercategory": "none", "id": 1, "name": "person"},
    {"supercategory": "none", "id": 2, "name": "car"},
    {"supercategory": "none", "id": 3, "name": "rider"},
    {"supercategory": "none", "id": 4, "name": "bus"},
    {"supercategory": "none", "id": 5, "name": "truck"},
    {"supercategory": "none", "id": 6, "name": "bike"},
    {"supercategory": "none", "id": 7, "name": "motor"},
    {"supercategory": "none", "id": 8, "name": "traffic light"},
    {"supercategory": "none", "id": 9, "name": "traffic sign"},
]
DET_NAME2ID = {c["name"]: c["id"] for c in DET_CATEGORIES}

DRIVABLE_CATEGORIES = [{"supercategory": "road", "id": 1, "name": "drivable"}]

LICENSES = [{"id": 1, "name": "CC BY-NC-SA 4.0",
             "url": "https://creativecommons.org/licenses/by-nc-sa/4.0/"}]

def info_block(title: str):
    return {"description": title, "url": "https://bdd-data.berkeley.edu/",
            "version": "1.0", "year": 2020, "contributor": "bdd2coco", "date_created": ""}

def resolve_image_path(split: str, file_name: str):
    p = os.path.join(IMAGE_ROOT_100K, split, file_name)
    return p if os.path.isfile(p) else None

def get_image_size(img_path: str, default=(1280, 720)):
    if img_path and os.path.isfile(img_path):
        try:
            with Image.open(img_path) as im:
                w, h = im.size
                return int(w), int(h)
        except Exception:
            pass
    return default

def clamp_point(x, y, w, h):
    return max(0.0, min(x, w - 1)), max(0.0, min(y, h - 1))

def has_closed_polygon(labels):
    if not labels: return False
    for l in labels:
        for poly in (l.get("poly2d") or []):
            if poly.get("closed", False) and len(poly.get("vertices", [])) >= 3:
                return True
    return False

def collect_closed_polys(labels):
    """Return list of flattened polygons (closed) for driveable; [] if none."""
    segs = []
    for l in labels:
        for poly in (l.get("poly2d") or []):
            if not poly.get("closed", False): 
                continue
            verts = poly.get("vertices", [])
            if len(verts) < 3: 
                continue
            segs.append(verts)  # keep as list of (x,y) for now
    return segs

def shoelace_area(pts):
    n = len(pts)
    if n < 3: return 0.0
    s = 0.0
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return abs(s) * 0.5

def bbox_from_pts(pts):
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]

def coco_skeleton(categories, title):
    return {"info": info_block(title), "licenses": LICENSES,
            "images": [], "annotations": [], "categories": categories, "type": "instances"}

def convert_split(split: str, src_json_path: str, out_det_path: str, out_drv_path: str):
    assert split in {"train", "val"}
    print(f"[{split}] Loading:", src_json_path)
    with open(src_json_path, "r", encoding="utf-8") as f:
        items = json.load(f)

    det = coco_skeleton(DET_CATEGORIES, f"BDD100K -> COCO detection (bbox-only), split={split}")
    drv = coco_skeleton(DRIVABLE_CATEGORIES, f"BDD100K -> COCO driveable polygons (closed), split={split}")

    ignored_det_categories = set()
    img_id = 0
    det_ann_id = 0
    drv_ann_id = 0

    kept, dropped = 0, 0
    masks_todo = []  # [(out_path, H, W, [flat polys])]

    for rec in tqdm(items, desc=f"[{split}] INTERSECTION filter"):
        file_name = rec.get("name")
        labels = rec.get("labels", []) or []

        # must have at least one closed polygon
        if not has_closed_polygon(labels):
            dropped += 1
            continue

        # must have at least one valid bbox (in selected classes)
        has_bbox = False
        for l in labels:
            cat = l.get("category")
            if not cat or cat not in DET_NAME2ID:
                if cat: ignored_det_categories.add(cat)
                continue
            box2d = l.get("box2d")
            if not box2d: 
                continue
            if (box2d["x2"] - box2d["x1"] > 0) and (box2d["y2"] - box2d["y1"] > 0):
                has_bbox = True
                break
        if not has_bbox:
            dropped += 1
            continue

        # Passed intersection filter -> keep image for BOTH outputs
        img_path = resolve_image_path(split, file_name)
        width, height = get_image_size(img_path)

        img_id += 1
        img_obj = {"file_name": file_name, "height": height, "width": width, "id": img_id}
        det["images"].append(img_obj)
        drv["images"].append(img_obj)

        # ---- Driveable polygons ----
        polys = collect_closed_polys(labels)
        flat_segs = []
        for verts in polys:
            # clamp + flatten
            clamped = [list(clamp_point(float(x), float(y), width, height)) for x, y in verts]
            flat = []
            for x, y in clamped:
                flat.extend([x, y])
            if len(flat) < 6:
                continue
            area = shoelace_area(clamped)
            bbox = bbox_from_pts(clamped)

            drv_ann_id += 1
            drv["annotations"].append({
                "id": drv_ann_id, "image_id": img_id,
                "category_id": 1, "iscrowd": 0,
                "segmentation": [flat], "area": float(area), "bbox": bbox
            })
            flat_segs.append(flat)

        # will write mask later
        if flat_segs:
            out_dir = MASK_DIR_TRAIN if split == "train" else MASK_DIR_VALID
            out_path = os.path.join(out_dir, os.path.splitext(file_name)[0] + ".png")
            masks_todo.append((out_path, height, width, flat_segs))

        # ---- Detection bboxes ----
        for l in labels:
            cat = l.get("category")
            if not cat or cat not in DET_NAME2ID:
                if cat: ignored_det_categories.add(cat)
                continue
            box2d = l.get("box2d")
            if not box2d:
                continue
            x1, y1 = float(box2d["x1"]), float(box2d["y1"])
            x2, y2 = float(box2d["x2"]), float(box2d["y2"])
            x1, y1 = clamp_point(x1, y1, width, height)
            x2, y2 = clamp_point(x2, y2, width, height)
            w, h = max(0.0, x2 - x1), max(0.0, y2 - y1)
            if w <= 0 or h <= 0:
                continue

            det_ann_id += 1
            det["annotations"].append({
                "id": det_ann_id, "image_id": img_id,
                "category_id": DET_NAME2ID[cat], "iscrowd": 0,
                "bbox": [x1, y1, w, h], "area": float(w * h)
            })

        kept += 1

    # Save COCO jsons
    with open(out_det_path, "w", encoding="utf-8") as f:
        json.dump(det, f)
    with open(out_drv_path, "w", encoding="utf-8") as f:
        json.dump(drv, f)

    print(f"[{split}] kept images (intersection): {kept} | dropped: {dropped}")
    print(f"[{split}] detection ignored categories: {sorted(list(ignored_det_categories))}")
    print(f"[{split}] saved -> {out_det_path}")
    print(f"[{split}] saved -> {out_drv_path}")

    # Write masks (binary 0/255)
    print(f"[{split}] writing masks...")
    for out_path, H, W, segs in tqdm(masks_todo, desc=f"[{split}] Masks"):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        mask = np.zeros((H, W), dtype=np.uint8)
        for seg in segs:
            pts = np.array(seg, dtype=np.int32).reshape((-1, 2))
            cv2.fillPoly(mask, [pts], 255)
        cv2.imwrite(out_path, mask)

def main():
    assert os.path.isfile(TRAIN_JSON), f"Missing: {TRAIN_JSON}"
    assert os.path.isfile(VAL_JSON),   f"Missing: {VAL_JSON}"

    print("=== START: Intersection COCO + Masks ===")
    print("LABELS DIR:", ROOT_LABELS)
    print("IMAGES ROOT:", IMAGE_ROOT_100K)
    print()

    convert_split("train", TRAIN_JSON, OUT_INST_TRAIN, OUT_DRV_TRAIN)
    convert_split("val",   VAL_JSON,   OUT_INST_VAL,   OUT_DRV_VAL)

    print("\nAll done.")

if __name__ == "__main__":
    main()