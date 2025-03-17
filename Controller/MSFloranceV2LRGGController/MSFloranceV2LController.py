import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import config as args
from pathlib import Path
import networkx as nx
import json
from tqdm import tqdm
from flask import Blueprint, request, jsonify, send_from_directory
import os
from flask_cors import CORS, cross_origin
import psycopg2

from transformers import AutoProcessor, AutoModelForCausalLM  
from PIL import Image
import requests
import copy
import os
import supervision as sv
from unittest.mock import patch
from transformers.dynamic_module_utils import get_imports

from florence_2 import utils

ms_floranceV2L_api = Blueprint('rgg', __name__)

def fixed_get_imports(filename: str | os.PathLike) -> list[str]:
    if not str(filename).endswith("modeling_florence2.py"):
        return get_imports(filename)
    imports = get_imports(filename)
    imports.remove("flash_attn")
    return imports

def load_model_ms_floranceV2_L():
    model_id = 'microsoft/Florence-2-large'
    with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports): #workaround for unnecessary flash_attn requirement
        model = AutoModelForCausalLM.from_pretrained(model_id, attn_implementation="sdpa", trust_remote_code=True)
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    
    return model, processor

@ms_floranceV2L_api.route('/rgg-gen', methods=['POST'])
@cross_origin()
def run():
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
        im = Image.open(file_name).convert("RGB")
        
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