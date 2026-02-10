import argparse

from datasets import load_dataset
from trl import GRPOTrainer, GRPOConfig
from trl.rewards import accuracy_reward

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm_server_host", type=str, default="", help="The server IP")
    args = parser.parse_args()

    dataset = load_dataset("trl-lib/DeepMath-103K", split="train")

    training_args = GRPOConfig(
        per_device_train_batch_size=4,
        use_vllm=True,
        vllm_server_host=args.vllm_server_host.replace("ip-", "").replace("-", "."),  # from ip-X-X-X-X to X.X.X.X
    )

    trainer = GRPOTrainer(
        model="Qwen/Qwen2.5-72B",
        args=training_args,
        reward_funcs=accuracy_reward,
        train_dataset=dataset
    )
    trainer.train()

if __name__=="__main__":
    main()