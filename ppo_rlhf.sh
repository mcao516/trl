# module load cudatoolkit/12.6
# conda activate trl

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

#  --total_episodes 100000 \
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file examples/accelerate_configs/deepspeed_zero2.yaml \
    examples/scripts/ppo/ppo_tldr.py \
        --dataset_name trl-lib/ultrafeedback_binarized  \
        --dataset_test_split test \
        --output_dir /home/mila/c/caomeng/scratch/trl/Qwen2.5-1.5B-ppo_double_score_${TIMESTAMP} \
        --learning_rate 3e-6 \
        --per_device_train_batch_size 2 \
        --gradient_accumulation_steps 16 \
        --response_length 512 \
        --num_train_epochs 1 \
        --model_name_or_path Qwen/Qwen2.5-1.5B \
        --sft_model_path Qwen/Qwen2.5-1.5B \
        --reward_model_path /home/mila/c/caomeng/scratch/trl/Qwen/Qwen/Qwen/Qwen3-0.6B-Reward-ultrafeedback \
        --local_rollout_forward_batch_size 16 \
        --missing_eos_penalty 1.0 \
        --stop_token eos \
        --eval_strategy steps \
        --eval_steps 10 \
        --report_to swanlab \
        --kl_coef 0.05 \
