#!/bin/bash
export PYTHONWARNINGS="ignore"

now=$(date +"%Y%m%d_%H%M%S")
model_name="StereoGAN"

CUDA_VISIBLE_DEVICES=0,1,2,3 python -u /code/test.py \
--print_freq=220 \
--output '/data/eval/final/stereogan_real_include' \
--use_multi_gpu=1 \
--maxdisp=192 \
--config-file '/code/configs/local_test_gan.yaml' \
--exclude-bg \
--load_from_mgpus_model 1 \
--load_dispnet_path '/dataset/eval/ep7_D1_0.4608_EPE6.6741.pth.rar' \
--load_gan_path '/dataset/eval/ep7_D1_0.4608_EPE6.6741.pth.rar' \
--load_checkpoints 1 \
--onReal
