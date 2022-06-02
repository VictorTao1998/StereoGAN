#!/bin/bash
export PYTHONWARNINGS="ignore"

now=$(date +"%Y%m%d_%H%M%S")
model_name="StereoGAN"

CUDA_VISIBLE_DEVICES=0,1,2,3 python -u /jianyu-fast-vol/StereoGAN/train_psm.py \
--model_type='dispnetc' \
--lr_rate=1e-5 \
--lr_gan=2e-5 \
--train_ratio_gan=3 \
--save_interval=1 \
--print_freq=50 \
--checkpoint_save_path="/jianyu-fast-vol/eval/StereoGAN_rui/checkpoints/${model_name}" \
--writer='/jianyu-fast-vol/eval/StereoGAN_rui' \
--use_multi_gpu=1 \
--maxdisp=192 \
--lambda_corr=1 \
--lambda_cycle=10 \
--lambda_id=5 \
--lambda_ms=0.1 \
--lambda_warp_inv=5 \
--lambda_disp_warp_inv=5 \
--config-file '/jianyu-fast-vol/StereoGAN/configs/remote_train_gan_v10.yaml' 
#--load_from_mgpus_model 1 \
#--load_dispnet_path '/jianyu-fast-vol/eval/StereoGAN_train_final/checkpoints/StereoGAN/ep1_D1_0.7821_EPE11.8211.pth.rar' \
#--load_gan_path '/jianyu-fast-vol/eval/StereoGAN_train_final/checkpoints/StereoGAN/ep1_D1_0.7821_EPE11.8211.pth.rar' \
#--load_checkpoints 1
