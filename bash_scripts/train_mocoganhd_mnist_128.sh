#!/bin/bash 

#SBATCH --job-name=mnistmocoganhd
#SBATCH --mem-per-cpu=2048
#SBATCH --partition long
#SBATCH --account cvit_bhaasha
#SBATCH --gres=gpu:4
#SBATCH --mincpus=38
#SBATCH --nodes=1
#SBATCH --time 10-00:00:00
#SBATCH --signal=B:HUP@600
#SBATCH -w gnode061

source /home2/aditya1/miniconda3/bin/activate mocogan

cd /ssd_scratch/cvit/aditya1/MoCoGAN-HD

CUDA_VISIBLE_DEVICES=0,1,2,3
python -W ignore train.py \
--name mnist --time_step 2 \
--lr 0.0001 \
--save_pca_path pca_stats/mnist \
--latent_dimension 512 \
--dataroot data/mnist_frames_128 \
--checkpoints_dir checkpoints/mnist \
--img_g_weights pretrained_stylegan2_models/mnist_network_snapshot.pt \
--multiprocessing_distributed \
--world_size 1 --rank 0 --batchSize 8 --workers 4 \
--style_gan_size 128 \
--total_epoch 200 \
--save_epoch_freq 1 \
--time_step 1