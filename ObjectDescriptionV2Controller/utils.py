import os
import numpy as np
import json
import torch
# from scipy.misc import imread, imresize
from PIL import Image
from tqdm import tqdm
from collections import Counter
from random import seed, choice, sample


def create_input_files(dataset, coco_json_path, image_folder, captions_per_image, min_word_freq, output_folder, max_len=100):
    """
    Creates input files for training, validation, and test data based on MSCOCO annotations.
    This version skips saving images to HDF5 and keeps the images in their original format on disk.

    :param dataset: name of dataset, usually 'coco'
    :param coco_json_path: path of MSCOCO JSON file with annotations and captions
    :param image_folder: folder with downloaded images
    :param captions_per_image: number of captions to sample per image
    :param min_word_freq: words occurring less frequently than this threshold are binned as <unk>s
    :param output_folder: folder to save files
    :param max_len: don't sample captions longer than this length
    """
    
    assert dataset == 'coco'

    # Read MSCOCO JSON annotations
    with open(coco_json_path, 'r') as j:
        data = json.load(j)

    # Read image paths and captions for each image
    image_paths = []
    image_captions = []
    image_ids = []  # List to store image_ids
    word_freq = Counter()

    # Create a mapping from image_id to captions
    image_id_to_captions = {}
    for ann in data['annotations']:
        image_id = ann['image_id']
        caption = ann['caption'].strip().split()
        if image_id not in image_id_to_captions:
            image_id_to_captions[image_id] = []
        if len(caption) <= max_len:
            image_id_to_captions[image_id].append(caption)
        word_freq.update(caption)

    # Process each image in the dataset
    for img in data['images']:
        image_id = img['id']
        if image_id not in image_id_to_captions:
            continue

        # Path to the image file
        path = os.path.join(image_folder, img['file_name'])

        image_paths.append(path)
        image_captions.append(image_id_to_captions[image_id])
        image_ids.append(image_id)  # Add the image_id to the list

    # Sanity check
    assert len(image_paths) == len(image_captions)

    # Create word map (vocabulary)
    words = [w for w in word_freq.keys() if word_freq[w] > min_word_freq]
    word_map = {k: v + 1 for v, k in enumerate(words)}
    word_map['<unk>'] = len(word_map) + 1
    word_map['<start>'] = len(word_map) + 1
    word_map['<end>'] = len(word_map) + 1
    word_map['<pad>'] = 0

    # Create a base/root name for all output files
    base_filename = dataset + '_' + str(captions_per_image) + '_cap_per_img_' + str(min_word_freq) + '_min_word_freq'

    # Save word map to a JSON
    with open(os.path.join(output_folder, 'WORDMAP_' + base_filename + '.json'), 'w') as j:
        json.dump(word_map, j)

    # Sample captions for each image, and save captions, their lengths, and image_ids to JSON files
    seed(123)
    for impaths, imcaps, img_ids, split in [(image_paths, image_captions, image_ids, 'TRAIN'),
                                            (image_paths, image_captions, image_ids, 'VAL'),
                                            (image_paths, image_captions, image_ids, 'TEST')]:
        enc_captions = []
        caplens = []
        img_paths_with_captions = []

        for i, path in enumerate(impaths):

            # Sample captions
            if len(imcaps[i]) < captions_per_image:
                captions = imcaps[i] + [choice(imcaps[i]) for _ in range(captions_per_image - len(imcaps[i]))]
            else:
                captions = sample(imcaps[i], k=captions_per_image)

            # Sanity check
            assert len(captions) == captions_per_image

            encoded_captions_per_image = []
            for j, c in enumerate(captions):
                # Encode captions
                enc_c = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in c] + [word_map['<end>']] + [word_map['<pad>']] * (max_len - len(c))

                # Find caption lengths
                c_len = len(c) + 2

                encoded_captions_per_image.append(enc_c)
                caplens.append(c_len)

            # Add image_id, image_path, and encoded captions for this image
            img_paths_with_captions.append({
                'image_id': img_ids[i],
                'image_path': path,
                'captions': encoded_captions_per_image
            })

        # Save encoded captions and image information to JSON
        with open(os.path.join(output_folder, split + '_CAPTIONS_' + base_filename + '.json'), 'w') as j:
            json.dump(img_paths_with_captions, j)

        # Save caption lengths to JSON
        with open(os.path.join(output_folder, split + '_CAPLENS_' + base_filename + '.json'), 'w') as j:
            json.dump(caplens, j)


def init_embedding(embeddings):
    """
    Fills embedding tensor with values from the uniform distribution.

    :param embeddings: embedding tensor
    """
    bias = np.sqrt(3.0 / embeddings.size(1))
    torch.nn.init.uniform_(embeddings, -bias, bias)


def load_embeddings(emb_file, word_map):
    """
    Creates an embedding tensor for the specified word map, for loading into the model.

    :param emb_file: file containing embeddings (stored in GloVe format)
    :param word_map: word map
    :return: embeddings in the same order as the words in the word map, dimension of embeddings
    """

    # Find embedding dimension
    with open(emb_file, 'r') as f:
        emb_dim = len(f.readline().split(' ')) - 1

    vocab = set(word_map.keys())

    # Create tensor to hold embeddings, initialize
    embeddings = torch.FloatTensor(len(vocab), emb_dim)
    init_embedding(embeddings)

    # Read embedding file
    print("\nLoading embeddings...")
    for line in open(emb_file, 'r'):
        line = line.split(' ')

        emb_word = line[0]
        embedding = list(map(lambda t: float(t), filter(lambda n: n and not n.isspace(), line[1:])))

        # Ignore word if not in train_vocab
        if emb_word not in vocab:
            continue

        embeddings[word_map[emb_word]] = torch.FloatTensor(embedding)

    return embeddings, emb_dim


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def save_checkpoint(data_name, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer, decoder_optimizer,
                    bleu4, is_best):
    """
    Saves model checkpoint.

    :param data_name: base name of processed dataset
    :param epoch: epoch number
    :param epochs_since_improvement: number of epochs since last improvement in BLEU-4 score
    :param encoder: encoder model
    :param decoder: decoder model
    :param encoder_optimizer: optimizer to update encoder's weights, if fine-tuning
    :param decoder_optimizer: optimizer to update decoder's weights
    :param bleu4: validation BLEU-4 score for this epoch
    :param is_best: is this checkpoint the best so far?
    """
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'bleu-4': bleu4,
             'encoder': encoder,
             'decoder': decoder,
             'encoder_optimizer': encoder_optimizer,
             'decoder_optimizer': decoder_optimizer}
    filename = 'checkpoint_' + data_name + '.pth.tar'
    torch.save(state, filename)
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        torch.save(state, 'BEST_' + filename)


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.

    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


def accuracy(scores, targets, k):
    """
    Computes top-k accuracy, from predicted and true labels.

    :param scores: scores from the model [batch_size * decode_lengths, vocab_size]
    :param targets: true labels [batch_size * decode_lengths]
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """

    batch_size = targets.size(0)  # batch_size * decode_lengths
    _, ind = scores.topk(k, 1, True, True)  # Get top-k predictions with shape [batch_size * decode_lengths, k]

    # Compare each element in targets with top-k predictions
    correct = (ind == targets.unsqueeze(1))  # Check if targets match any of the top-k predictions

    # Calculate total correct predictions
    correct_total = correct.view(-1).float().sum()
    return correct_total.item() * (100.0 / batch_size)



