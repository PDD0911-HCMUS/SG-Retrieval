from transformers import AutoProcessor, AutoModelForCausalLM  
from PIL import Image, ImageDraw, ImageFont
import requests
import copy
import os
import supervision as sv
from unittest.mock import patch
from transformers.dynamic_module_utils import get_imports
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from florence_2 import utils

def fixed_get_imports(filename: str | os.PathLike) -> list[str]:
    if not str(filename).endswith("modeling_florence2.py"):
        return get_imports(filename)
    imports = get_imports(filename)
    imports.remove("flash_attn")
    return imports

model_id = 'microsoft/Florence-2-large'
with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports): #workaround for unnecessary flash_attn requirement
    model = AutoModelForCausalLM.from_pretrained(model_id, attn_implementation="sdpa", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)


def run_infer(task_promt, image, text_input = None):
    if(text_input is None):
        promt = task_promt

    else:
        promt = task_promt + text_input
    
    inputs = processor(text=promt, images=image, return_tensors="pt")
    generated_ids = model.generate(
        input_ids = inputs["input_ids"],
        pixel_values = inputs["pixel_values"],
        max_new_tokens = 1024,
        early_stopping = False,
        do_sample = False,
        num_beams = 3
    )

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text,
        task = task_promt,
        image_size=(image.width, image.height)
    )

    return parsed_answer

if __name__ == "__main__":
    vg_image_dir = "/home/duypd/ThisPC-DuyPC/SG-Retrieval/0_Datasets/VisualGenome/VG_100K/"
    image_id = "235.jpg"
    image = Image.open(vg_image_dir + image_id).convert("RGB")

    utils.set_model_info(model, processor)

    tasks = [utils.TaskType.CAPTION,
            utils.TaskType.DETAILED_CAPTION,
            utils.TaskType.MORE_DETAILED_CAPTION,
            utils.TaskType.DENSE_REGION_CAPTION,
            utils.TaskType.REGION_TO_DESCRIPTION]
    
    ##################################################
    # for task in [utils.TaskType.PHRASE_GROUNDING]: #
    #     results = utils.run_example(task, image)   #
    #     print(f'{task.value}{results[task]}')      #
    #     utils.plot_bbox(results[task], image)      #
    ##################################################

    # task_prompt = utils.TaskType.DENSE_REGION_CAPTION
    # results = utils.run_example(task_prompt, image)
    # print(results['<DENSE_REGION_CAPTION>'].keys())
    # utils.plot_bbox(results[task_prompt], image)

    # task_prompt = utils.TaskType.MORE_DETAILED_CAPTION
    # results = utils.run_example(task_prompt, image)
    # print(results)
    # # utils.plot_bbox(results[task_prompt], image)

    # Get a caption
    task_prompt = utils.TaskType.MORE_DETAILED_CAPTION
    results = utils.run_example(task_prompt, image)

    # Use the output as the input into the next task (phrase grounding)
    text_input = results[task_prompt]
    task_prompt = utils.TaskType.PHRASE_GROUNDING
    results = utils.run_example(task_prompt, image, text_input)

    results[utils.TaskType.MORE_DETAILED_CAPTION] = text_input

    lst_text = text_input.split('.')
    print(lst_text)
    print(len(lst_text))
    print(len(results[utils.TaskType.PHRASE_GROUNDING]['bboxes']))

    data = results[utils.TaskType.PHRASE_GROUNDING]
    boxes = data['bboxes'][:len(lst_text)]
    labels = data['labels'][:len(lst_text)]

    fig, ax = plt.subplots()

    # Display the image
    ax.imshow(image)

    # Plot each bounding box
    for bbox, label in zip(boxes, labels):
        # Unpack the bounding box coordinates
        x1, y1, x2, y2 = bbox
        # Create a Rectangle patch
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='r', facecolor='none')
        # Add the rectangle to the Axes
        ax.add_patch(rect)
        # Annotate the label
        plt.text(x1, y1, label, color='white', fontsize=8, bbox=dict(facecolor='red', alpha=0.5))

    # Remove the axis ticks and labels
    ax.axis('off')

    # Show the plot
    plt.show()