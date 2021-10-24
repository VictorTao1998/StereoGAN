#!/bin/bash
export PYTHONWARNINGS="ignore"

now=$(date +"%Y%m%d_%H%M%S")
model_name="StereoGAN"

CUDA_VISIBLE_DEVICES=0,1,2,3 python -u /jianyu-fast-vol/StereoGAN/test.py \
--print_freq=220 \
--output '/jianyu-fast-vol/eval/StereoGAN_test' \
--use_multi_gpu=1 \
--maxdisp=192 \
--config-file '/jianyu-fast-vol/StereoGAN/configs/remote_test_gan_v10.yaml' \
--exclude-bg \
--exclude-zeros \
--load_from_mgpus_model 1 \
--load_dispnet_path '/jianyu-fast-vol/eval/StereoGAN_train/checkpoints/StereoGAN/ep2_D1_0.2991_EPE5.4978.pth.rar' \
--load_gan_path '/jianyu-fast-vol/eval/StereoGAN_train/checkpoints/StereoGAN/ep2_D1_0.2991_EPE5.4978.pth.rar' \
--load_checkpoints 1
