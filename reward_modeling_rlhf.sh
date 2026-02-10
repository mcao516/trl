#!/bin/bash
# SBATCH --nodes=5
# SBATCH --gres=gpu:8
# module python/3.10
module load cudatoolkit/12.6
# source ~/envTRL/bin/activate

# accelerate launch --config_file examples/accelerate_configs/deepspeed_zero2.yaml \
# python examples/scripts/reward_modeling.py \

accelerate launch --config_file examples/accelerate_configs/deepspeed_zero3.yaml \
  examples/scripts/reward_modeling.py \
    --model_name_or_path Qwen/Qwen3-0.6B \
    --dataset_name trl-lib/ultrafeedback_binarized \
    --output_dir /home/mila/c/caomeng/scratch/trl/Qwen/Qwen/Qwen/Qwen3-0.6B-Reward-ultrafeedback \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 1 \
    --learning_rate 1.0e-4 \
    --eval_strategy steps \
    --eval_steps 50 \
    --max_length 2048 \
    --report_to swanlab

# python examples/scripts/reward_modeling.py \
#     --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
#     --dataset_name trl-lib/ultrafeedback_binarized \
#     --output_dir /home/mila/c/caomeng/scratch/trl/Qwen2-0.5B-Reward-LoRA \
#     --per_device_train_batch_size 8 \
#     --num_train_epochs 1 \
#     --learning_rate 1.0e-4 \
#     --eval_strategy steps \
#     --eval_steps 50 \
#     --max_length 2048 \
#     --use_peft \
#     --lora_task_type SEQ_CLS \
#     --lora_r 32 \
#     --lora_alpha 16 \
#     --report_to swanlab