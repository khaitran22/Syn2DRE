#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=250G
#SBATCH --qos=gpu
#SBATCH --job-name=testing
#SBATCH --time=6:00:00
#SBATCH --partition=gpu_cuda
#SBATCH --gres=gpu:1
#SBATCH --account=a_xue_li
#SBATCH --mail-type=ALL
#SBATCH --constraint=cuda80gb
#SBATCH --mail-user=phankhai.tran@uq.edu.au
#SBATCH -o /scratch/user/uqptran9/Code/Work_3/dev/Fine-tuning-stage/slurm.output
#SBATCH -e /scratch/user/uqptran9/Code/Work_3/dev/Fine-tuning-stage/slurm.error

module load anaconda3/2022.05
source /home/uqptran9/.bashrc
conda activate /home/uqptran9/.conda/envs/conda-pytorch-212

srun /home/uqptran9/.conda/envs/conda-pytorch-212/bin/python3.11 /scratch/user/uqptran9/Code/Work_3/dev/Fine-tuning-stage/main.py \
 --data_dir /scratch/user/uqptran9/Code/Work_3/dev/dataset/re_docred/ \
 --transformer_type bert \
 --model_name_or_path bert-base-cased \
 --train_file train_revised.json \
 --dev_file dev_revised.json \
 --test_file test_revised.json \
 --train_distant_file train_distant.json \
 --num_labels 5 \
 --main2_lr 3e-5 \
 --main2_classifier_lr 1e-4 \
 --main1_lr 5e-5 \
 --main1_classifier_lr 1e-4 \
 --gamma_pos 0.5 \
 --gamma_neg 0.5 \
 --train_batch_size 8 \
 --distant_batch_size 16 \
 --test_batch_size 8 \
 --num_train_epochs 20.0 \
 --save_path_main_1 /scratch/user/uqptran9/Code/Work_3/dev/Fine-tuning-stage/checkpoints/student1.pt \
 --save_path_main_2 /scratch/user/uqptran9/Code/Work_3/dev/Fine-tuning-stage/checkpoints/student2.pt \
 --save_path_teacher_1 /scratch/user/uqptran9/Code/Work_3/dev/Fine-tuning-stage/checkpoints/teacher1.pt \
 --save_path_teacher_2 /scratch/user/uqptran9/Code/Work_3/dev/Fine-tuning-stage/checkpoints/teacher2.pt \
 --load_path_main_1 /scratch/user/uqptran9/Code/Work_3/baselines/UGDRE/checkpoints/best/UGDRE-RE-pretrain1_66.45_teacher1.pt \
 --load_path_main_2 /scratch/user/uqptran9/Code/Work_3/dev/KD-DocRE/chkpoints/best/model_ptr_1_66.45_teacher1.pt \
 --warmup_epochs 15 \
 --drop_prob 0.1 \
 --dropout 0.25 \
 --agree_weight 0.1 \
 --cross_update_step 6 \
 --cross_update \
 --self_training