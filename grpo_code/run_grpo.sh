#!/bin/bash
source ~/envTRL/bin/activate

CUDA_VISIBLE_DEVICES=0,1,2,3 ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file accelerate_configs/zero3.yaml \
    --num_processes 4 \
    run_grpo.py \