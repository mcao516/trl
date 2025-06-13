# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import shutil

import torch
import wandb
from accelerate import PartialState
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
)

from trl import (
    ModelConfig,
    PPOConfig,
    PPOTrainer,
    ScriptArguments,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE
from trl.core import LengthSampler


def resize_score_layer(model, bias_value=None):
    in_features = model.score.in_features
    use_bias = bias_value is not None and abs(bias_value) > 1e-5 

    original_layer = model.score
    original_weights = original_layer.weight.data  # Shape: [2, 768]
    weights_for_label_1 = original_weights[1:2, :]
    new_score_layer = torch.nn.Linear(in_features, 1, bias=use_bias)
    with torch.no_grad():
        new_score_layer.weight.copy_(weights_for_label_1)
        if use_bias:
            new_score_layer.bias.fill_(bias_value)
    model.score = new_score_layer
    model.config.num_labels = 1


if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, PPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_into_dataclasses()
    # remove output_dir if exists
    shutil.rmtree(training_args.output_dir, ignore_errors=True)

    state = PartialState()
    if state.is_main_process:
        wandb.init(**{"project": "trl_ppo_imdb", "name": training_args.exp_name})
    state.wait_for_everyone()
    ################
    # Model & Tokenizer
    ################
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, padding_side="left", trust_remote_code=model_args.trust_remote_code
    )
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    value_model = AutoModelForSequenceClassification.from_pretrained(
        training_args.reward_model_path, trust_remote_code=model_args.trust_remote_code, num_labels=2
    )
    # new_score_layer = torch.nn.Linear(value_model.score.in_features, 1, bias=False)
    # value_model.score = new_score_layer
    resize_score_layer(value_model, bias_value=3.6)
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        training_args.reward_model_path, trust_remote_code=model_args.trust_remote_code, num_labels=2
    )
    reward_model.resize_token_embeddings(len(tokenizer))
    reward_model.config.pad_token_id = tokenizer.pad_token_id
    # resize_score_layer(reward_model, bias_value=3.6)
    # tokenizer.pad_token = tokenizer.eos_token
    # reward_model.config.pad_token_id = reward_model.config.eos_token_id
    policy = AutoModelForCausalLM.from_pretrained(
        training_args.sft_model_path, trust_remote_code=model_args.trust_remote_code
    )

    peft_config = get_peft_config(model_args)
    if peft_config is None:
        ref_policy = AutoModelForCausalLM.from_pretrained(
            training_args.sft_model_path, trust_remote_code=model_args.trust_remote_code
        )
    else:
        ref_policy = None

    ################
    # Dataset
    ################
    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
    train_dataset = dataset[script_args.dataset_train_split]
    eval_dataset = dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None

    def prepare_dataset(dataset, tokenizer, input_min_text_length=2, input_max_text_length=8):
        """pre-tokenize the dataset before training; only collate during training"""

        input_size = LengthSampler(input_min_text_length, input_max_text_length)

        def tokenize(element):
            input_ids = tokenizer.encode(
                element["text"],
                padding=False,
            )[:input_size()]
            return {"input_ids": input_ids, "lengths": len(input_ids)}

        return dataset.map(
            tokenize,
            remove_columns=dataset.column_names,
            num_proc=training_args.dataset_num_proc,
        )

    # Compute that only on the main process for faster data processing.
    # see: https://github.com/huggingface/trl/pull/1255
    with PartialState().local_main_process_first():
        train_dataset = prepare_dataset(train_dataset, tokenizer)
        if eval_dataset is not None:
            eval_dataset = prepare_dataset(eval_dataset, tokenizer)

    assert train_dataset[0]["input_ids"][-1] != tokenizer.eos_token_id, "The last token should not be an EOS token"
    ################
    # Training
    ################
    trainer = PPOTrainer(
        args=training_args,
        processing_class=tokenizer,
        model=policy,
        ref_model=ref_policy,
        reward_model=reward_model,
        value_model=value_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)

    trainer.generate_completions()
