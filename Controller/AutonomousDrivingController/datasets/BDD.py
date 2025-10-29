#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
check_coco_image_consistency.py

Kiểm tra nhanh tính đồng nhất ảnh giữa 2 file COCO JSON:
- So sánh tập file_name giữa A và B
- Kiểm tra trùng lặp file_name trong mỗi file
- Kiểm tra width/height có khớp giữa 2 file cho từng file_name chung
- Kiểm tra annotation có image_id hợp lệ (tồn tại trong "images")
- Trả về mã thoát 0 nếu đồng nhất hoàn toàn, 1 nếu có sai khác

Cách dùng:
    python check_coco_image_consistency.py \
        --a /path/to/instances_train2017.json \
        --b /path/to/driveable_train2017.json \
        [--max-show 20]
"""

import json
import argparse
from collections import Counter, defaultdict
import sys

def load_coco(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    images = data.get("images", [])
    anns = data.get("annotations", [])
    return images, anns

def index_images(images):
    # map: file_name -> (id, width, height)
    name2meta = {}
    dup_names = []
    for im in images:
        fn = im.get("file_name")
        if fn in name2meta:
            dup_names.append(fn)
        name2meta[fn] = (im.get("id"), im.get("width"), im.get("height"))
    return name2meta, dup_names

def check_ann_refs(anns, valid_img_ids):
    bad = []
    for a in anns:
        iid = a.get("image_id")
        if iid not in valid_img_ids:
            bad.append(a.get("id"))
    return bad

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="COCO JSON A (ví dụ: instances_train2017.json)")
    ap.add_argument("--b", required=True, help="COCO JSON B (ví dụ: driveable_train2017.json)")
    ap.add_argument("--max-show", type=int, default=20, help="Giới hạn số phần tử in ra khi lệch")
    args = ap.parse_args()

    images_a, anns_a = load_coco(args.a)
    images_b, anns_b = load_coco(args.b)

    # index images
    name2meta_a, dup_a = index_images(images_a)
    name2meta_b, dup_b = index_images(images_b)

    set_a = set(name2meta_a.keys())
    set_b = set(name2meta_b.keys())

    only_a = sorted(list(set_a - set_b))
    only_b = sorted(list(set_b - set_a))
    inter  = sorted(list(set_a & set_b))

    print("== SUMMARY ==")
    print(f"A: {args.a}")
    print(f"B: {args.b}")
    print(f"- num images A: {len(set_a)} (raw entries: {len(images_a)})")
    print(f"- num images B: {len(set_b)} (raw entries: {len(images_b)})")
    print(f"- intersection : {len(inter)}")
    print(f"- only in A    : {len(only_a)}")
    print(f"- only in B    : {len(only_b)}")

    if dup_a:
        print(f"\n[WARNING] Duplicate file_name in A ({len(dup_a)}). Examples:")
        for x in dup_a[:args.max_show]:
            print("  -", x)

    if dup_b:
        print(f"\n[WARNING] Duplicate file_name in B ({len(dup_b)}). Examples:")
        for x in dup_b[:args.max_show]:
            print("  -", x)

    if only_a:
        print(f"\n[DIFF] Present only in A (show up to {args.max_show}):")
        for x in only_a[:args.max_show]:
            print("  -", x)

    if only_b:
        print(f"\n[DIFF] Present only in B (show up to {args.max_show}):")
        for x in only_b[:args.max_show]:
            print("  -", x)

    # Check size mismatch for common images
    size_mismatch = []
    for fn in inter:
        ida, wa, ha = name2meta_a[fn]
        idb, wb, hb = name2meta_b[fn]
        if (wa != wb) or (ha != hb):
            size_mismatch.append((fn, (wa, ha), (wb, hb)))
    if size_mismatch:
        print(f"\n[DIFF] Image size mismatch on {len(size_mismatch)} files (show up to {args.max_show}):")
        for fn, (wa, ha), (wb, hb) in size_mismatch[:args.max_show]:
            print(f"  - {fn}: A=({wa}x{ha}), B=({wb}x{hb})")

    # Check that annotations reference valid image_ids
    valid_ids_a = set([im.get("id") for im in images_a])
    valid_ids_b = set([im.get("id") for im in images_b])

    bad_ann_a = check_ann_refs(anns_a, valid_ids_a)
    bad_ann_b = check_ann_refs(anns_b, valid_ids_b)

    if bad_ann_a:
        print(f"\n[ERROR] {len(bad_ann_a)} annotations in A reference unknown image_id. Examples:")
        for x in bad_ann_a[:args.max_show]:
            print("  - ann_id:", x)

    if bad_ann_b:
        print(f"\n[ERROR] {len(bad_ann_b)} annotations in B reference unknown image_id. Examples:")
        for x in bad_ann_b[:args.max_show]:
            print("  - ann_id:", x)

    # Final verdict
    ok = (len(only_a) == 0 and
          len(only_b) == 0 and
          len(size_mismatch) == 0 and
          len(dup_a) == 0 and
          len(dup_b) == 0 and
          len(bad_ann_a) == 0 and
          len(bad_ann_b) == 0)

    print("\n== RESULT ==")
    if ok:
        print("✔ Two COCO files are CONSISTENT on images.")
        sys.exit(0)
    else:
        print("✖ Inconsistencies detected. See details above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
