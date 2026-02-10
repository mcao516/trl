# module load cudatoolkit/12.6
# conda activate trl

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")


accelerate launch --config_file examples/accelerate_configs/deepspeed_zero2.yaml \
    examples/scripts/ppo/ppo_tldr.py \
        --dataset_name trl-lib/tldr \
        --dataset_test_split validation \
        --output_dir pythia-1b-deduped-tldr-preference-sft-trl-style-ppo \
        --learning_rate 3e-6 \
        --per_device_train_batch_size 32 \
        --gradient_accumulation_steps 2 \
        --total_episodes 1000000 \
        --model_name_or_path EleutherAI/pythia-1b-deduped \
        --sft_model_path cleanrl/EleutherAI_pythia-1b-deduped__sft__tldr \
        --reward_model_path cleanrl/EleutherAI_pythia-1b-deduped__reward__tldr \
        --local_rollout_forward_batch_size 16 \
        --missing_eos_penalty 1.0 \
        --stop_token eos \
        --eval_strategy steps \
        --eval_steps 20 \
        --report_to wandb \