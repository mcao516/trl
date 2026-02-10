
import argparse
from pathlib import Path
from typing import Optional, List
import re
import os


import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig  
from peft import LoraConfig, PeftModel
from trl import GRPOConfig, GRPOTrainer

from paths import CACHE_DIR, TRAIN_DATA_DIR, PROC_DATA_DIR

SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""

cot_instructions = """Let's think step by step, you MUST write the answer as an integer after '####' without including the units. Write the answer at the end.\n\n"""

coe_instructions = f"Let's think step by step with minimal natural language explanation, you MUST write the answer as an integer after '####' without including the units. Write the answer at the end.\n\n"


COE_FEW_SHOT_EXAMPLES = """\
Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
Answer: 48/2 = 24, 48+24 = 72, #### 72

Question: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?
Answer: 12/60 = $0.2, 0.2 x 50 = $10, #### 10
"""

COT_FEW_SHOT_EXAMPLES = """\
Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
Answer: Natalia sold 48/2 = 24 clips in May.
Natalia sold 48+24 = 72 clips altogether in April and May.
#### 72

Question: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?
Answer: Weng earns 12/60 = $0.2 per minute.
Working 50 minutes, she earned 0.2 x 50 = $10.
#### 10
"""

INSTRUCTION_MAP = {
    "coe": coe_instructions,
    "cot": cot_instructions,
}

FEW_SHOT_MAP =  {
    "coe": COE_FEW_SHOT_EXAMPLES,
    "cot": COT_FEW_SHOT_EXAMPLES,
}


def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip().replace(",", "").replace("$", "")


# uncomment middle messages for 2-shot prompting
def get_gsm8k_questions(split = "train", dataset_name= "gsm8k", dataset_type = "cot", lora_checkpoint=None) -> Dataset:

    data = load_dataset("csv", data_files=f"{PROC_DATA_DIR}/{dataset_name}_train.csv")['train'] # type: ignore
    
    def _map(x):
        if lora_checkpoint:
            prompt = f"{INSTRUCTION_MAP[dataset_type]}\nQuestion: {x['problem']}\nAnswer: "
        else:
            prompt = f"{INSTRUCTION_MAP[dataset_type]}{FEW_SHOT_MAP[dataset_type]}\nQuestion: {x['problem']}\nAnswer: "
        return {
            "prompt": prompt,
            "answer": x["answer"],  # <-- critical
        }
    data = data.map(_map)  # type: ignore
    
    return data  # type: ignore
    

# Reward functions
def format_reward_func(completions, **kwargs):
    #pattern = r"\n#### \d+"    
    pattern = r"####\s*[\d\.]+"
    completion_contents = [completion.split("Question:")[0] for completion in completions]    
    matches = [re.search(pattern, content) for content in completion_contents]
    print([0.5 if match else 0.0 for match in matches])
    return [0.5 if match else 0.0 for match in matches]

def correctness_reward_func(completions, answer, **kwargs):
    rewards = []
    
    for completion, ground_truth in zip(completions, answer) :
        try:
            completion = completion.split("Question:")[0]
            print("Completion:", completion)
            match = re.search(r'####.*?([\d,]+(?:\.\d+)?)', completion)
            if match:
                parsed_answer = match.group(1)
                
                for remove_char in [',', '$', '%', 'g']:
                    parsed_answer = parsed_answer.replace(remove_char, '')
                    
                if abs(float(parsed_answer)-float(ground_truth)) < 1e-3:
                    rewards.append(1.0)
                else:
                    rewards.append(0.0)
                
            else:
                rewards.append(0.0)
        
        except ValueError:
            rewards.append(0.0)
    
    print('-'*20, f"\nGround Truth Answer:\n{answer[0]}", f"\nSampled Response:\n{completions[0]}")
    return rewards


    
# def load_training_config():
#     training_args = GRPOConfig(
#         output_dir=output_dir,
#         run_name=run_name,
#         learning_rate=2e-6,
#         adam_beta1 = 0.9,
#         adam_beta2 = 0.99,
#         weight_decay = 0.1,
#         warmup_ratio = 0.1,
#         lr_scheduler_type='cosine',
#         logging_steps=1,
#         bf16=True,
#         per_device_train_batch_size=1,
#         gradient_accumulation_steps=8,
#         num_generations=8,
#         max_prompt_length=512,
#         max_completion_length=256,
#         num_train_epochs=1,
#         save_steps=100,
#         max_grad_norm=0.1,
#         report_to="wandb",
#         log_on_each_node=False,
#         #use_vllm=True,
#         #vllm_device='auto',
#     #     generation_kwargs={
#     #        "stop_strings": ["Question:"],
#     #         #"tokenizer": tokenizer   # <-- add this
#     #     }
#     )
#     return training_args

def build_lora_config(
    r: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
) -> LoraConfig:
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        task_type="CAUSAL_LM",
    )

# def load_peft_config():
#     peft_config = LoraConfig(
#         r=16,
#         lora_alpha=32,
#         target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
#         task_type="CAUSAL_LM",
#         lora_dropout=0.05,
#     )
#     return lora_config

def load_model_and_tokenizer(
    base_model: str,
    sft_lora_checkpoint: Optional[str],
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=None,
        cache_dir=str(CACHE_DIR),
    )
    
    if sft_lora_checkpoint:
        model = PeftModel.from_pretrained(model, sft_lora_checkpoint, is_trainable=True) # make sure that the lora is trainable

    model.to("cuda")
    model.config.use_cache = False  # recommended for training

    tok = AutoTokenizer.from_pretrained(base_model, cache_dir=str(CACHE_DIR))
    tok.pad_token = tok.eos_token
    tok.padding_side = "right" # is this the right side for training?
    
    return model, tok


# def load_model_and_tokenizer(base_model_name, lora_checkpoint=None):
#     model = AutoModelForCausalLM.from_pretrained(
#         base_model_name,
#         torch_dtype=torch.bfloat16,
#         #quantization_config=bnb_config,      # Apply 8-bit quantization
#         attn_implementation="flash_attention_2",
#         device_map=None,
#         cache_dir = CACHE_DIR,
#     ).to("cuda")
    
#     if lora_checkpoint:
#         # Path to your LoRA SFT checkpoint
#         model = PeftModel.from_pretrained(model, lora_checkpoint)
    
#     tokenizer = AutoTokenizer.from_pretrained(base_model_name)
#     tokenizer.pad_token = tokenizer.eos_token

#     return model, tokenizer


def make_run_name(model_name: str, dataset_name: str, ft_type: str, dataset_type: str, sft_lora_checkpoint: str) -> str:
    mn = model_name.split("/")[-1]
    if not sft_lora_checkpoint:
        mn += "-base"
    return f"{mn}-{dataset_name}-{dataset_type}-{ft_type}-GRPO"

def find_latest_checkpoint(output_dir: str) -> Optional[str]:
    p = Path(output_dir)
    if not p.exists():
        return None
    ckpts = [d for d in p.iterdir() if d.is_dir() and d.name.startswith("checkpoint")]
    if not ckpts:
        return None
    ckpts.sort(key=lambda x: x.stat().st_mtime)
    return str(ckpts[-1])





def main():
    ap = argparse.ArgumentParser()

    # core
    ap.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-1B")
    ap.add_argument("--dataset_name", type=str, default="gsm8k")
    ap.add_argument("--dataset_type", type=str, choices=["coe", "cot"], default="coe")

    # starting point (optional): load an existing SFT LoRA adapter into base model before GRPO
    ap.add_argument("--sft_lora_checkpoint", type=str, default=None)


    # output / resume
    ap.add_argument("--output_root", type=str, default="models/")
    ap.add_argument("--resume", action="store_true", default=True)

    # GRPOConfig
    ap.add_argument("--lr", type=float, default=2e-6)
    ap.add_argument("--adam_beta1", type=float, default=0.9)
    ap.add_argument("--adam_beta2", type=float, default=0.99)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--lr_scheduler_type", type=str, default="cosine")
    ap.add_argument("--logging_steps", type=int, default=1)
    ap.add_argument("--bf16", action="store_true", default=False)
    ap.add_argument("--per_device_train_batch_size", type=int, default=1)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=8)
    ap.add_argument("--num_generations", type=int, default=8)
    ap.add_argument("--max_prompt_length", type=int, default=256)
    ap.add_argument("--max_completion_length", type=int, default=256)
    ap.add_argument("--num_train_epochs", type=int, default=1)
    ap.add_argument("--save_steps", type=int, default=100)
    ap.add_argument("--max_grad_norm", type=float, default=0.1)
    ap.add_argument("--report_to", type=str, default="wandb")
    ap.add_argument("--log_on_each_node", action="store_true", default=False)

    # GRPO LoRA (peft_config passed to GRPOTrainer)
    ap.add_argument("--use_lora", action="store_true", default=True)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)

    args = ap.parse_args()

    # get dataset
    dataset = get_gsm8k_questions(dataset_type=args.dataset_type, dataset_name = args.dataset_name, lora_checkpoint=args.sft_lora_checkpoint)

    # model / tokenizer (optional: load SFT LoRA into base)
    model, tokenizer = load_model_and_tokenizer(
        base_model=args.base_model,
        sft_lora_checkpoint=args.sft_lora_checkpoint,
    )
    
    # check that only lora parameters can be trained
    trainable = [n for n,p in model.named_parameters() if p.requires_grad]
    print("num trainable", len(trainable))
    print(trainable[:20])

    # run name / output dir
    ft_type = "lora"
    if (not args.sft_lora_checkpoint) and (not args.use_lora):
        ft_type = "full"

    run_name = make_run_name(args.base_model, args.dataset_name, ft_type, args.dataset_type, args.sft_lora_checkpoint)
    output_dir = str(Path(args.output_root) / run_name)
    os.makedirs(output_dir, exist_ok=True)

    # training config
    training_args = GRPOConfig(
        output_dir=output_dir,
        run_name=run_name,
        learning_rate=args.lr,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=args.logging_steps,
        bf16=args.bf16,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        num_train_epochs=args.num_train_epochs,
        save_steps=args.save_steps,
        max_grad_norm=args.max_grad_norm,
        report_to=args.report_to,
        log_on_each_node=args.log_on_each_node,
    )

    peft_config = None
    if args.use_lora and (not args.sft_lora_checkpoint):
        peft_config = build_lora_config(
            r=args.lora_r,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
        )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[format_reward_func, correctness_reward_func],
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config, # continue training on sft lora adpater weights
    )

    resume_ckpt = None
    if args.resume:
        resume_ckpt = find_latest_checkpoint(output_dir)
        if resume_ckpt is not None:
            print(f"Resuming from checkpoint: {resume_ckpt}")

    trainer.train(resume_from_checkpoint=resume_ckpt if resume_ckpt is not None else None)


if __name__ == "__main__":
    main()