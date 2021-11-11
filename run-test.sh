#!/bin/bash
export PYTHONWARNINGS="ignore"

now=$(date +"%Y%m%d_%H%M%S")
model_name="StereoGAN"

CUDA_VISIBLE_DEVICES=0,1,2,3 python -u /jianyu-fast-vol/StereoGAN/test.py \
--print_freq=220 \
--output '/jianyu-fast-vol/eval/StereoGAN_test_f_real_exclude' \
--use_multi_gpu=1 \
--maxdisp=192 \
--config-file '/jianyu-fast-vol/StereoGAN/configs/remote_test_gan_v10.yaml' \
--exclude-bg \
--load_from_mgpus_model 1 \
--load_dispnet_path '/jianyu-fast-vol/eval/StereoGAN_train_final_1/checkpoints/StereoGAN/ep4_D1_0.5868_EPE8.5989.pth.rar' \
--load_gan_path '/jianyu-fast-vol/eval/StereoGAN_train_final_1/checkpoints/StereoGAN/ep4_D1_0.5868_EPE8.5989.pth.rar' \
--load_checkpoints 1 \
--exclude-zeros \
--onReal
