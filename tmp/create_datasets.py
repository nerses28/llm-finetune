import os
import json
from datasets import load_dataset


dataset = load_dataset("BramVanroy/ultra_feedback_dutch_cleaned", "sft_gpt4_hq")


with open("datasets/ultrachat/train.jsonl", "w+") as f:
    for i, example in enumerate(dataset["train_sft"]):
        messages = example["messages"]
        f.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")


with open("datasets/ultrachat/test.jsonl", "w+") as f:
    for i, example in enumerate(dataset["test_sft"]):
        messages = example["messages"]
        f.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")


dataset = load_dataset("BramVanroy/ultra_feedback_dutch_cleaned", "dpo_hq")


with open("datasets/ultrachat_dpo/train.jsonl", "w+") as f:
    for i, example in enumerate(dataset["train_prefs"]):
        prompt = example["prompt"]
        prompt = [{"content": example["prompt"], "role": "user"}]
        chosen = example["chosen"]
        rejected = example["rejected"]
        f.write(json.dumps({"prompt": prompt, "chosen": chosen, "rejected": rejected}, ensure_ascii=False) + "\n")


with open("datasets/ultrachat_dpo/test.jsonl", "w+") as f:
    for i, example in enumerate(dataset["test_prefs"]):
        prompt = example["prompt"]
        prompt = [{"content": example["prompt"], "role": "user"}]
        chosen = example["chosen"]
        rejected = example["rejected"]
        f.write(json.dumps({"prompt": prompt, "chosen": chosen, "rejected": rejected}, ensure_ascii=False) + "\n")