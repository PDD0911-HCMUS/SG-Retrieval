import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
from modelsRelTR.backbone import Backbone, Joiner
from modelsRelTR.position_encoding import PositionEmbeddingSine
from modelsRelTR.transformer import Transformer
from modelsRelTR.reltr import RelTR
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import ConfigArgs as args
from transformers.dynamic_module_utils import get_imports
from unittest.mock import patch
from transformers import BlipProcessor, BlipForConditionalGeneration

CLASSES = args.CLASSES
REL_CLASSES = args.REL_CLASSES
transform = T.Compose([
    T.Resize(512),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def create_model_reltr():
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

def create_model_blip():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
    return model, processor

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

def do_infer(image_path):
    im = Image.open(image_path)
    img = transform(im).unsqueeze(0)
    # propagate through the model
    model = create_model_reltr()
    model_blip, processor = create_model_blip()
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

    for i, (idx, (sxmin, symin, sxmax, symax), (oxmin, oymin, oxmax, oymax)) in \
                enumerate(zip(keep_queries, sub_bboxes_scaled[indices], obj_bboxes_scaled[indices])):
        
        box_s = (sxmin, symin, sxmax, symax)
        box_s = tuple(map(int, box_s))

        box_o = (oxmin, oymin, oxmax, oymax)
        box_o = tuple(map(int, box_o))

        print(box_s)

        cropped_image_s = im.crop(box_s)
        cropped_image_o = im.crop(box_o)

        cropped_image_s = processor(images=cropped_image_s, return_tensors="pt")
        cropped_image_o = processor(images=cropped_image_o, return_tensors="pt")

        # cropped_image_s.show()
        # cropped_image_o.show()

        output_s = model_blip.generate(**cropped_image_s)
        output_o = model_blip.generate(**cropped_image_o)

        caption_s = processor.decode(output_s[0], skip_special_tokens=True)
        caption_o = processor.decode(output_o[0], skip_special_tokens=True)

        print(f"caption_s: {caption_s}")
        print(f"caption_o: {caption_o}")
            


if __name__ == "__main__":
    img_path = args.img_folder_vg + '235.jpg'
    do_infer(image_path=img_path)

    #/tmp/tmptxn1irvx.PNG