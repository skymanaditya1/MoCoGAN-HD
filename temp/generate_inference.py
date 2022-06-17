# method used for generating the data using the pretrained checkpoints 
import re
from glob import glob
import os
import os.path as osp
from tqdm import tqdm
import argparse

RESULTS_DIR = '/ssd_scratch/cvit/aditya1/mocoganhd_results'
MOCOGANHD_DIR = '/ssd_scratch/cvit/aditya1/MoCoGAN-HD'

def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)

def choose_k(checkpoint_paths, k=10):
    # checkpoints_sampled = [checkpoint_paths[i] for i in range(len(checkpoint_paths)) if i%k == 0]
    checkpoints_sampled = [checkpoint_paths[i] for i in range(1, len(checkpoint_paths), k)]
    return checkpoints_sampled

def generate_result(gpu_id, pca_paths, img_weights_path, checkpoint_dir, epoch, results_dir, num_videos=2048):
#     command_format = 'CUDA_VISIBLE_DEVICES={} python -W ignore evaluate.py  \
#   --save_pca_path {} \
#   --latent_dimension 512 \
#   --style_gan_size {} \
#   --img_g_weights {} \
#   --load_pretrain_path {} \
#   --load_pretrain_epoch {} \
#   --results_dir {} \
#   --num_test_videos {} \ '

    command_format = 'CUDA_VISIBLE_DEVICES={} \
        python -W ignore evaluate.py \
        --save_pca_path {} \
        --latent_dimension 512 --style_gan_size 128 \
        --img_g_weights {} --load_pretrain_path {} \
        --load_pretrain_epoch {} --results {} --num_test_videos {}'

    command = command_format.format(gpu_id, pca_paths, img_weights_path, checkpoint_dir, epoch, results_dir, num_videos)

    # command = command_format.format(gpu_id, pca_paths, stylegan_size, img_weights_path, checkpoint_dir, epoch, results_dir, num_videos)

    print(f'Executing command : {command}')

    output = os.popen(command).read()

    print(output)

# code to get the epoch id from the checkpoint path
def get_epoch_from_checkpoint(checkpoint_path):
    filename = osp.basename(checkpoint_path)
    epoch_id = filename.rsplit('_', 1)[1].split('.')[0]

    return epoch_id

def main(args):

    os.chdir(MOCOGANHD_DIR)

    checkpoint_dir = args.checkpoint_dir
    pca_path = args.pca_path
    img_generator_weights = args.generator_weights
    dataset = args.dataset

    # the results dir should be constructed from the checkpoint path

    num_videos = args.num_videos

    gpu_id = 2
    # stylegan_size = 128


    files = glob(checkpoint_dir + '/modelD_img_epoch*')
    sort_nicely(files)
    
    # the value of k should be such that only 20% of the dataset is used -- not that 
    # k = int(0.2 * len(files)) 
    # to use only 20% of the dataset -- k can be set to 5
    k = 8

    sampled = choose_k(files, k)

    # for the sampled checkpoints, fetch the epoch number and use the epoch number for inferring mocoganhd
    print(f'Generating results for the pretrained checkpoints : {len(sampled)}')

    for checkpoint in tqdm(sampled):
        epoch = get_epoch_from_checkpoint(checkpoint)

        # construct the results dir from the checkpoint path 
        results_dir = osp.join(RESULTS_DIR, dataset, osp.basename(checkpoint).split('.')[0])
        os.makedirs(results_dir, exist_ok=True)

        print(f'Epoch : {epoch}')

        generate_result(gpu_id, pca_path, img_generator_weights, checkpoint_dir, epoch, results_dir, num_videos)        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--pca_path', type=str)
    parser.add_argument('--generator_weights', type=str)
    parser.add_argument('--checkpoint_dir', type=str)
    # parser.add_argument('--results_dir', type=str)
    parser.add_argument('--num_videos', type=int, default=2048)
    parser.add_argument('--gpu_id', type=int)
    args = parser.parse_args()

    # checkpoint_dir = '/ssd_scratch/cvit/aditya1/MoCoGAN-HD/checkpoints/skytimelapse/skytimelapse_22_06_06_07_02_42'
    main(args)