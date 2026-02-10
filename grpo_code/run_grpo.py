from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from trl.rewards import think_format_reward, reasoning_accuracy_reward
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig

dataset_name = 'AI-MO/NuminaMath-TIR'
train_dataset = load_dataset(dataset_name, split='train[:5%]')

print(train_dataset)
print(train_dataset[0])

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant  "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process is enclosed strictly within <think> and </think> tags. "
    "After closing </think>, the assistant MUST provide the final answer in plain text."
)


def make_conversation(example):
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["problem"]},
        ],
    }

train_dataset = train_dataset.map(make_conversation)
print(train_dataset[0]['prompt'])
train_dataset = train_dataset.remove_columns(['messages', 'problem'])
print(train_dataset)

model_id, output_dir = "Qwen/Qwen3-8B", "/home/mila/c/caomeng/scratch/trl/models"   



model = AutoModelForCausalLM.from_pretrained(
    model_id,
    attn_implementation="sdpa",                   # Change to Flash Attention if GPU has support
    dtype="bfloat16",                              # Change to bfloat16 if GPU has support
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,                        # Load the model in 4-bit precision to save memory
        bnb_4bit_compute_dtype=torch.float16,     # Data type used for internal computations in quantization
        bnb_4bit_use_double_quant=True,           # Use double quantization to improve accuracy
        bnb_4bit_quant_type="nf4"                 # Type of quantization. "nf4" is recommended for recent LLMs
    )
)


# You may need to update `target_modules` depending on the architecture of your chosen model.
# For example, different LLMs might have different attention/projection layer names.
peft_config = LoraConfig(
    r=32,
    lora_alpha=32,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",],
)

# Configure training arguments using GRPOConfig
training_args = GRPOConfig(
    # Training schedule / optimization
    learning_rate=2e-5,                                     # Learning rate for the optimizer
    #num_train_epochs=1,
    max_steps=500,                                          # Number of dataset passes. For full trainings, use `num_train_epochs` instead

    # Parameters that control GRPO training (you can adapt them)
    per_device_train_batch_size = 2,
    max_completion_length=256, # default: 256               # Max completion length produced during training
    num_generations=2, # default: 8                         # Number of generations produced during trainig for comparison

    # Optimizations
    optim = "paged_adamw_8bit",                             # Optimizer
    use_liger_kernel=True,                                  # Enable Liger kernel optimizations for faster training

    # Parameters related to reporting and saving
    output_dir=output_dir,                                  # Where to save model checkpoints and logs
    logging_steps=1,                                        # Log training metrics every N steps
    report_to="trackio",                                    # Experiment tracking tool
    trackio_space_id="Qwen/Qwen3-8B-trackio",               # HF Space where the experiment tracking will be saved
    # Hub integration
    push_to_hub=False,                                      # Automatically push the trained model to the Hugging Face Hub
                                                            # The model will be saved under your Hub account in the repository named `output_dir`
    # vLLM params
    # use_vllm=True,                                        # Activate vLLM training for faster training
    # vllm_mode='colocate',
    # vllm_gpu_memory_utilization=0.3,
    # vllm_enable_sleep_mode=True
)

print("Start training...")
trainer = GRPOTrainer(
    model=model,
    reward_funcs=[think_format_reward, reasoning_accuracy_reward],
    args=training_args,
    train_dataset=train_dataset,
    peft_config=peft_config,
)

gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)

print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

trainer_stats = trainer.train()

used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)

print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

trainer.save_model(output_dir)