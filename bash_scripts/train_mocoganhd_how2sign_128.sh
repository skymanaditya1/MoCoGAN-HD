#!/bin/bash 

#SBATCH --job-name=h2s_mocoganhd
#SBATCH --mem-per-cpu=2048
#SBATCH --partition long
#SBATCH --account cvit_bhaasha
#SBATCH --gres=gpu:4
#SBATCH --mincpus=38
#SBATCH --nodes=1
#SBATCH --time 10-00:00:00
#SBATCH --signal=B:HUP@600
#SBATCH -w gnode075

source /home2/aditya1/miniconda3/bin/activate mocogan
cd /ssd_scratch/cvit/aditya1/MoCoGAN-HD
CUDA_VISIBLE_DEVICES=0,1 \
python -W ignore train.py \
--name how2sign_faces_128 \
--lr 0.0001 \
--save_pca_path pca_stats/how2sign_faces_128 \
--latent_dimension 512 \
--dataroot ../mocoganhd_training_data/how2sign_faces_styleganv_resized \
--checkpoints_dir checkpoints/how2sign_faces_128 \
--img_g_weights ../stylegan2_checkpoints/how2sign_faces_stylegan2_009600.pt \
--multiprocessing_distributed --world_size 1 --rank 0 \
--batchSize 4 --workers 4 --style_gan_size 128 \
--total_epoch 200 --save_epoch_freq 1 --time_step 1