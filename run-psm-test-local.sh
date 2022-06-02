#!/bin/bash
export PYTHONWARNINGS="ignore"

now=$(date +"%Y%m%d_%H%M%S")
model_name="StereoGAN"

CUDA_VISIBLE_DEVICES=0,1,2,3 python -u /code/test_psm.py \
--print_freq=220 \
--output '/dataset/eval/stereogan/StereoGAN_rui' \
--use_multi_gpu=1 \
--maxdisp=192 \
--config-file '/code/configs/local_test_gan.yaml' \
--exclude-bg \
--load_from_mgpus_model 1 \
--load_dispnet_path '/dataset/eval/stereogan/ep2_D1_0.2991_EPE5.4978.pth.rar' \
--load_gan_path '/dataset/eval/stereogan/ep2_D1_0.2991_EPE5.4978.pth.rar' \
--load_checkpoints 1 