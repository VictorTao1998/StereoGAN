#!/bin/bash
export PYTHONWARNINGS="ignore"

now=$(date +"%Y%m%d_%H%M%S")
model_name="StereoGAN"

CUDA_VISIBLE_DEVICES=0,1,2,3 python -u test.py \
--print_freq=220 \
--output '/data/eval/StereoGAN_test' \
--use_multi_gpu=1 \
--maxdisp=192 \
--config-file '/code/configs/local_test_gan.yaml' \
--exclude-bg \
--exclude-zeros \
--load_from_mgpus_model 1 \
--load_dispnet_path '/data/pretrain/ep1_D1_0.5722_EPE7.7740.pth.rar' \
--load_gan_path '/data/pretrain/ep1_D1_0.5722_EPE7.7740.pth.rar' \
--load_checkpoints 1
