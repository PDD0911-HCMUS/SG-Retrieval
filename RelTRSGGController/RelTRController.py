import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
from .models.backbone import Backbone, Joiner
from .models.position_encoding import PositionEmbeddingSine
from .models.transformer import Transformer
from .models.reltr import RelTR
import ConfigArgs as args
from pathlib import Path
import networkx as nx
import json
from tqdm import tqdm
from flask import Blueprint, request, jsonify, send_from_directory
import os
from flask_cors import CORS, cross_origin
import psycopg2

sgg_api = Blueprint('sgg', __name__)


CLASSES = args.CLASSES
REL_CLASSES = args.REL_CLASSES
transform = T.Compose([
    T.Resize(512),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def create_model():
    position_embedding = PositionEmbeddingSine(128, normalize=True)
    backbone = Backbone('resnet50', False, False, False)
    backbone = Joiner(backbone, position_embedding)
    backbone.num_channels = 2048

    transformer = Transformer(d_model=256, dropout=0.1, nhead=8,
                            dim_feedforward=2048,
                            num_encoder_layers=6,
                            num_decoder_layers=6,
                            normalize_before=False,
                            return_intermediate_dec=True)

    model = RelTR(backbone, transformer, num_classes=151, num_rel_classes = 51,
                num_entities=100, num_triplets=200)

    # The checkpoint is pretrained on Visual Genome
    ckpt = torch.load(args.ckpt_sgg, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    model.eval()

    return model


# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
          (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def draw_sg(sList, oList, rList, out_res_grp):
    G = nx.DiGraph()
    for s,o,r in zip(sList,oList,rList):
        G.add_edge(s, o, label=r)

    # Vẽ đồ thị
    pos = nx.shell_layout(G)  # Sắp xếp các node

    # Vẽ các node và cạnh
    nx.draw(G, pos, with_labels=True, node_size=1000, node_color='skyblue', font_size=10, font_weight='bold', arrowsize=15)
    
    # Vẽ các nhãn cho cạnh
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=10)

    plt.savefig(out_res_grp)
    plt.close()

def draw_single_object_dect(im, indices, 
                            keep_queries, 
                            sub_bboxes_scaled, 
                            obj_bboxes_scaled,
                            out_res_obj):
    fig, axs = plt.subplots(nrows=1, figsize=(16, 9))
    axs.imshow(im)
    for idx, (sxmin, symin, sxmax, symax), (oxmin, oymin, oxmax, oymax) in \
                zip(keep_queries, sub_bboxes_scaled[indices], obj_bboxes_scaled[indices]):
            
        # ax.imshow(im)
        axs.axis('off')
        axs.add_patch(plt.Rectangle((sxmin, symin), sxmax - sxmin, symax - symin,
                                    fill=False, color='blue', linewidth=2.5))
        axs.add_patch(plt.Rectangle((oxmin, oymin), oxmax - oxmin, oymax - oymin,
                                    fill=False, color='orange', linewidth=2.5))
    
    fig.savefig(out_res_obj, pad_inches=0)
    plt.close(fig)
    return

def draw_per_detail_triplet(im, indices, 
                            keep_queries, 
                            sub_bboxes_scaled, 
                            obj_bboxes_scaled,
                            out_res_obj):
    
    for i, (idx, (sxmin, symin, sxmax, symax), (oxmin, oymin, oxmax, oymax)) in \
                enumerate(zip(keep_queries, sub_bboxes_scaled[indices], obj_bboxes_scaled[indices])):
            
        fig, axs = plt.subplots(nrows=1, figsize=(16, 9))
        axs.imshow(im)
        axs.axis('off')
        axs.add_patch(plt.Rectangle((sxmin, symin), sxmax - sxmin, symax - symin,
                                    fill=False, color='blue', linewidth=2.5))
        axs.add_patch(plt.Rectangle((oxmin, oymin), oxmax - oxmin, oymax - oymin,
                                    fill=False, color='orange', linewidth=2.5))
        out_res_obj_per = f"{out_res_obj}_{i}.png"
        fig.savefig(out_res_obj_per, pad_inches=0)
        plt.close(fig)
    return

def draw_detail_triplet(im, indices, 
                        keep_queries, 
                        sub_bboxes_scaled, 
                        obj_bboxes_scaled,
                        probas_sub,
                        probas,
                        probas_obj,
                        out_triplet):
    num_images = len(indices)
    rows = 2
    # if(num_images > 4):
    #     rows = 3
    # else:
    #     rows = 2
    fig, axs = plt.subplots(ncols=(num_images + rows - 1) // rows, nrows=rows, figsize=(22, 5))
    axs = axs.flatten() if rows > 1 else [axs]
    for ax in axs:
        ax.axis('off') 
    for idx, ax_i, (sxmin, symin, sxmax, symax), (oxmin, oymin, oxmax, oymax) in \
            zip(keep_queries, axs, sub_bboxes_scaled[indices], obj_bboxes_scaled[indices]):
        
        ax_i.imshow(im)
        ax_i.add_patch(plt.Rectangle((sxmin, symin), sxmax - sxmin, symax - symin,
                                    fill=False, color='blue', linewidth=2.5))
        ax_i.add_patch(plt.Rectangle((oxmin, oymin), oxmax - oxmin, oymax - oymin,
                                    fill=False, color='orange', linewidth=2.5))

        ax_i.axis('off')
        ax_i.set_title(CLASSES[probas_sub[idx].argmax()]+' '+REL_CLASSES[probas[idx].argmax()]+' '+CLASSES[probas_obj[idx].argmax()], fontsize=10)
    
    fig.savefig(out_triplet, pad_inches=0)
    plt.close(fig)
    return

def extract_triplet(mode):
    if(mode == 'train'):
        annfile = args.anno_train
        with open(annfile) as f:
            anno = json.load(f)
    if(mode == 'val'):
        annfile = args.anno_valid
        with open(annfile) as f:
            anno = json.load(f)

    model = create_model()
    data_jsons = []
    for item in tqdm(anno['images'][:]):
        try:
            im = Image.open(args.img_folder_vg + item['file_name'])
            img = transform(im).unsqueeze(0)
            outputs = model(img)

            # keep only predictions with >0.3 confidence
            probas = outputs['rel_logits'].softmax(-1)[0, :, :-1]
            probas_sub = outputs['sub_logits'].softmax(-1)[0, :, :-1]
            probas_obj = outputs['obj_logits'].softmax(-1)[0, :, :-1]
            keep = torch.logical_and(probas.max(-1).values > 0.7, torch.logical_and(probas_sub.max(-1).values > 0.7,
                                                                                    probas_obj.max(-1).values > 0.7))
            topk = 10 # display up to 10 images
            keep_queries = torch.nonzero(keep, as_tuple=True)[0]
            indices = torch.argsort(-probas[keep_queries].max(-1)[0] * probas_sub[keep_queries].max(-1)[0] * probas_obj[keep_queries].max(-1)[0])[:topk]
            keep_queries = keep_queries[indices]

            # save the attention weights
            conv_features, dec_attn_weights_sub, dec_attn_weights_obj = [], [], []
            hooks = [
                model.backbone[-2].register_forward_hook(
                    lambda self, input, output: conv_features.append(output)
                ),
                model.transformer.decoder.layers[-1].cross_attn_sub.register_forward_hook(
                    lambda self, input, output: dec_attn_weights_sub.append(output[1])
                ),
                model.transformer.decoder.layers[-1].cross_attn_obj.register_forward_hook(
                    lambda self, input, output: dec_attn_weights_obj.append(output[1])
                )]

            with torch.no_grad():
                outputs = model(img)

                for hook in hooks:
                    hook.remove()
                # don't need the list anymore
                conv_features = conv_features[0]
                dec_attn_weights_sub = dec_attn_weights_sub[0]
                dec_attn_weights_obj = dec_attn_weights_obj[0]
                sList, oList, rList, t = [], [], [], []
                
                for idx in keep_queries:
                    sList.append(CLASSES[probas_sub[idx].argmax()])
                    oList.append(CLASSES[probas_obj[idx].argmax()])
                    rList.append(REL_CLASSES[probas[idx].argmax()])
                    txt = CLASSES[probas_sub[idx].argmax()] + ' ' + REL_CLASSES[probas[idx].argmax()] + ' ' + CLASSES[probas_obj[idx].argmax()]
                    t.append(txt)
                
                data_json = {
                    "image_id": item['file_name'],
                    "triplet": t
                }

                data_jsons.append(data_json)
        except:
            continue

    with open(mode + '_trip.json', 'w') as f:
        json.dump(data_jsons, f)
    return data_jsons

@sgg_api.route('/sgg-gen', methods=['POST'])
@cross_origin()
def sgg_controller():
    try:
        if 'file' not in request.files:
            return jsonify(
                Data = None,
                Status = False, 
                Msg = 'No file part in the request'
                )

        file = request.files['file']

        if file.filename == '':
            return jsonify(
                Data = None,
                Status = False, 
                Msg = 'No selected file'
            )
        if file:
            filepath = os.path.join(args.dir_upload, file.filename)
            file.save(filepath)

        fileName = file.filename
        file_name = args.dir_upload + fileName
        path = Path(file_name.replace('.jpg', ''))
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
        im = Image.open(file_name)
        img = transform(im).unsqueeze(0)
        # propagate through the model
        model = create_model()
        outputs = model(img)

        # keep only predictions with >0.3 confidence
        probas = outputs['rel_logits'].softmax(-1)[0, :, :-1]
        probas_sub = outputs['sub_logits'].softmax(-1)[0, :, :-1]
        probas_obj = outputs['obj_logits'].softmax(-1)[0, :, :-1]
        threshhold = 0.3
        keep = torch.logical_and(probas.max(-1).values > threshhold, torch.logical_and(probas_sub.max(-1).values > threshhold,
                                                                                probas_obj.max(-1).values > threshhold))

        # convert boxes from [0; 1] to image scales
        sub_bboxes_scaled = rescale_bboxes(outputs['sub_boxes'][0, keep], im.size)
        obj_bboxes_scaled = rescale_bboxes(outputs['obj_boxes'][0, keep], im.size)

        topk = 10 # display up to 10 images
        keep_queries = torch.nonzero(keep, as_tuple=True)[0]
        indices = torch.argsort(-probas[keep_queries].max(-1)[0] * probas_sub[keep_queries].max(-1)[0] * probas_obj[keep_queries].max(-1)[0])[:topk]
        keep_queries = keep_queries[indices]

        # save the attention weights
        conv_features, dec_attn_weights_sub, dec_attn_weights_obj = [], [], []
        hooks = [
            model.backbone[-2].register_forward_hook(
                lambda self, input, output: conv_features.append(output)
            ),
            model.transformer.decoder.layers[-1].cross_attn_sub.register_forward_hook(
                lambda self, input, output: dec_attn_weights_sub.append(output[1])
            ),
            model.transformer.decoder.layers[-1].cross_attn_obj.register_forward_hook(
                lambda self, input, output: dec_attn_weights_obj.append(output[1])
            )]

        with torch.no_grad():
            # propagate through the model
            outputs = model(img)

            for hook in hooks:
                hook.remove()

            # don't need the list anymore
            conv_features = conv_features[0]
            dec_attn_weights_sub = dec_attn_weights_sub[0]
            dec_attn_weights_obj = dec_attn_weights_obj[0]

            sList, oList, rList, t, t_obj = [], [], [], [], []
            
            for idx in keep_queries:
                sList.append(CLASSES[probas_sub[idx].argmax()])
                oList.append(CLASSES[probas_obj[idx].argmax()])
                rList.append(REL_CLASSES[probas[idx].argmax()])
                txt = CLASSES[probas_sub[idx].argmax()] + ' ' + REL_CLASSES[probas[idx].argmax()] + ' ' + CLASSES[probas_obj[idx].argmax()]

                t_json = {
                    "subject": CLASSES[probas_sub[idx].argmax()],
                    "relation": REL_CLASSES[probas[idx].argmax()],
                    "object": CLASSES[probas_obj[idx].argmax()]
                }
                t.append(txt)
                t_obj.append(t_json)

            out_res_grp = str(path)+'/'+args.prefix_graph+fileName
            out_res_obj = str(path)+'/'+args.prefix_name+fileName
            out_triplet = str(path)+'/'+args.prefix_triplet+fileName

            draw_single_object_dect(im, indices, keep_queries, sub_bboxes_scaled, obj_bboxes_scaled, out_res_obj)
            draw_per_detail_triplet(im, indices, keep_queries, sub_bboxes_scaled, obj_bboxes_scaled, out_res_obj)
            draw_detail_triplet(im, indices, keep_queries, sub_bboxes_scaled, obj_bboxes_scaled, probas_sub, probas, probas_obj, out_triplet)
            draw_sg(sList,oList,rList, out_res_grp)

        # return [args.prefix_name+fileName, args.prefix_graph+fileName, args.prefix_triplet+fileName, fileName, t]
        res = {
            'objectDet': args.prefix_name+fileName,
            'graphDet': args.prefix_graph+fileName,
            'tripletDet': args.prefix_triplet+fileName,
            'triplets': t,
            'tripletSet': t_obj 
        }

        return jsonify(
            Data = res,
            Status = True, 
            Msg = 'Scene Graph Generated Successfully'
        )
    except Exception as e:
        return jsonify(
            Data = None,
            Status = False, 
            Msg = str(e)
        )

def insert_sgg():
    conn = psycopg2.connect(args.conn_str)
    cursor = conn.cursor()
    model = create_model()
    vg_images = os.listdir(args.img_folder_vg)
    sql = """INSERT INTO "Image2GraphEmbedding_V2" ("image_name","triplets") VALUES(%s,%s);"""
    for file_name in tqdm(vg_images):
        try:
            im = Image.open(args.img_folder_vg + file_name)
            img = transform(im).unsqueeze(0)
            # propagate through the model
            outputs = model(img)
            # keep only predictions with >0.3 confidence
            probas = outputs['rel_logits'].softmax(-1)[0, :, :-1]
            probas_sub = outputs['sub_logits'].softmax(-1)[0, :, :-1]
            probas_obj = outputs['obj_logits'].softmax(-1)[0, :, :-1]
            threshhold = 0.3
            keep = torch.logical_and(probas.max(-1).values > threshhold, torch.logical_and(probas_sub.max(-1).values > threshhold,
                                                                                    probas_obj.max(-1).values > threshhold))
            topk = 10 # display up to 10 images
            keep_queries = torch.nonzero(keep, as_tuple=True)[0]
            indices = torch.argsort(-probas[keep_queries].max(-1)[0] * probas_sub[keep_queries].max(-1)[0] * probas_obj[keep_queries].max(-1)[0])[:topk]
            keep_queries = keep_queries[indices]
            # save the attention weights
            conv_features, dec_attn_weights_sub, dec_attn_weights_obj = [], [], []
            hooks = [
                model.backbone[-2].register_forward_hook(
                    lambda self, input, output: conv_features.append(output)
                ),
                model.transformer.decoder.layers[-1].cross_attn_sub.register_forward_hook(
                    lambda self, input, output: dec_attn_weights_sub.append(output[1])
                ),
                model.transformer.decoder.layers[-1].cross_attn_obj.register_forward_hook(
                    lambda self, input, output: dec_attn_weights_obj.append(output[1])
                )]
            with torch.no_grad():
                # propagate through the model
                outputs = model(img)
                for hook in hooks:
                    hook.remove()
                # don't need the list anymore
                conv_features = conv_features[0]
                dec_attn_weights_sub = dec_attn_weights_sub[0]
                dec_attn_weights_obj = dec_attn_weights_obj[0]
                t = []
                for idx in keep_queries:
                    txt = CLASSES[probas_sub[idx].argmax()] + ' ' + REL_CLASSES[probas[idx].argmax()] + ' ' + CLASSES[probas_obj[idx].argmax()]
                    t.append(txt)

                if(len(t) >= 7):
                    cursor.execute(sql,(file_name ,t) )
                    conn.commit()
        except Exception as e:
            continue
            # break

def insert_sgg_mscoco():
    conn = psycopg2.connect(args.conn_str)
    cursor = conn.cursor()
    model = create_model()
    mscoco_images = os.listdir(args.img_folder_mscoco)
    sql = """INSERT INTO "Image2GraphEmbedding_V2_MSCOCO" ("image_name","triplets") VALUES(%s,%s);"""
    for file_name in tqdm(mscoco_images):
        try:
            im = Image.open(args.img_folder_mscoco + file_name)
            img = transform(im).unsqueeze(0)
            # propagate through the model
            outputs = model(img)
            # keep only predictions with >0.3 confidence
            probas = outputs['rel_logits'].softmax(-1)[0, :, :-1]
            probas_sub = outputs['sub_logits'].softmax(-1)[0, :, :-1]
            probas_obj = outputs['obj_logits'].softmax(-1)[0, :, :-1]
            threshhold = 0.3
            keep = torch.logical_and(probas.max(-1).values > threshhold, torch.logical_and(probas_sub.max(-1).values > threshhold,
                                                                                    probas_obj.max(-1).values > threshhold))
            topk = 10 # display up to 10 images
            keep_queries = torch.nonzero(keep, as_tuple=True)[0]
            indices = torch.argsort(-probas[keep_queries].max(-1)[0] * probas_sub[keep_queries].max(-1)[0] * probas_obj[keep_queries].max(-1)[0])[:topk]
            keep_queries = keep_queries[indices]
            # save the attention weights
            conv_features, dec_attn_weights_sub, dec_attn_weights_obj = [], [], []
            hooks = [
                model.backbone[-2].register_forward_hook(
                    lambda self, input, output: conv_features.append(output)
                ),
                model.transformer.decoder.layers[-1].cross_attn_sub.register_forward_hook(
                    lambda self, input, output: dec_attn_weights_sub.append(output[1])
                ),
                model.transformer.decoder.layers[-1].cross_attn_obj.register_forward_hook(
                    lambda self, input, output: dec_attn_weights_obj.append(output[1])
                )]
            with torch.no_grad():
                # propagate through the model
                outputs = model(img)
                for hook in hooks:
                    hook.remove()
                # don't need the list anymore
                conv_features = conv_features[0]
                dec_attn_weights_sub = dec_attn_weights_sub[0]
                dec_attn_weights_obj = dec_attn_weights_obj[0]
                t = []
                for idx in keep_queries:
                    txt = CLASSES[probas_sub[idx].argmax()] + ' ' + REL_CLASSES[probas[idx].argmax()] + ' ' + CLASSES[probas_obj[idx].argmax()]
                    t.append(txt)

                if(len(t) >= 7):
                    cursor.execute(sql,(file_name ,t) )
                    conn.commit()
        except Exception as e:
            continue
        # break

@sgg_api.route('/res-sgg/<filename>')
def upload_image(filename):
    if('object+' in filename):
        return send_from_directory(args.dir_upload + filename.replace('.jpg', '').replace('object+', ''), filename)
    if('graph+' in filename):
        return send_from_directory(args.dir_upload + filename.replace('.jpg', '').replace('graph+', ''), filename)
    if('triplet+' in filename):
        return send_from_directory(args.dir_upload + filename.replace('.jpg', '').replace('triplet+', ''), filename)
    
if __name__ == "__main__":
    insert_sgg_mscoco()