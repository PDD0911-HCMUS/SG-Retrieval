from utils import create_input_files

if __name__ == '__main__':
    # Create input files (along with word map)
    create_input_files(dataset='coco',
                       coco_json_path='/home/duypd/ThisPC-DuyPC/SG-Retrieval/Datasets/VisualGenome/anno_org/caption.json',
                       image_folder='/home/duypd/ThisPC-DuyPC/SG-Retrieval/Datasets/VisualGenome/VG_100K_cropped',
                       captions_per_image=1,
                       min_word_freq=5,
                       output_folder='/home/duypd/ThisPC-DuyPC/SG-Retrieval/Datasets/VisualGenome/anno_org',
                       max_len=20)