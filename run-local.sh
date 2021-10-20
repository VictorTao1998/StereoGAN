#!/bin/bash
export PYTHONWARNINGS="ignore"

now=$(date +"%Y%m%d_%H%M%S")
model_name="StereoGAN"

CUDA_VISIBLE_DEVICES=0,1,2,3 python -u train.py \
--model_type='dispnetc' \
--lr_rate=1e-5 \
--lr_gan=2e-5 \
--train_ratio_gan=3 \
--save_interval=5 \
--print_freq=10 \
--checkpoint_save_path="/data/eval/StereoGAN_train/checkpoints/${model_name}" \
--writer='/data/eval/StereoGAN_train' \
--use_multi_gpu=1 \
--maxdisp=192 \
--lambda_corr=1 \
--lambda_cycle=10 \
--lambda_id=5 \
--lambda_ms=0.1 \
--lambda_warp_inv=5 \
--lambda_disp_warp_inv=5 \
--config-file '/code/configs/local_train_gan_v10.yaml'
