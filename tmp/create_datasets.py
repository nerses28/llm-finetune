import os
import json
from datasets import load_dataset


dataset = load_dataset("smangrul/ultrachat-10k-chatml")


with open("datasets/no_system_prompt/train.jsonl", "w+") as f:
    for i, example in enumerate(dataset["train"]):
        messages = example["messages"]
        f.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
        if i > 1000:
            break


with open("datasets/no_system_prompt/validation.jsonl", "w+") as f:
    for i, example in enumerate(dataset["test"]):
        messages = example["messages"]
        f.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
        if i > 1000:
            break


with open("datasets/system_prompt/train.jsonl", "w+") as f:
    for i, example in enumerate(dataset["train"]):
        messages = [{"role": "system", "content": "You are a helpful AI assistant."}] + example["messages"]
        f.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
        if i > 1000:
            break


with open("datasets/system_prompt/validation.jsonl", "w+") as f:
    for i, example in enumerate(dataset["test"]):
        messages = [{"role": "system", "content": "You are a helpful AI assistant."}] + example["messages"]
        f.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
        if i > 1000:
            break
