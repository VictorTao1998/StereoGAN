#!/bin/bash
export PYTHONWARNINGS="ignore"

now=$(date +"%Y%m%d_%H%M%S")
model_name="StereoGAN"

CUDA_VISIBLE_DEVICES=0,1,2,3 python -u /code/StereoGAN/test_psm.py \
--print_freq=220 \
--output '/media/jianyu/dataset/eval/stereogan/eval_local_sim' \
--use_multi_gpu=1 \
--maxdisp=192 \
--config-file '/code/StereoGAN/configs/local_test_gan.yaml' \
--exclude-bg \
--load_from_mgpus_model 1 \
--load_dispnet_path '/media/jianyu/dataset/eval/stereogan/train_prim_nautilus/ep2_D1_0.2256_EPE4.6078.pth.rar' \
--load_gan_path '/media/jianyu/dataset/eval/stereogan/train_prim_nautilus/ep2_D1_0.2256_EPE4.6078.pth.rar' \
--load_checkpoints 1 \
--onReal