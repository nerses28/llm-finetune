import os
from enum import Enum

import torch
from datasets import DatasetDict, load_dataset, load_from_disk
from datasets.builder import DatasetGenerationError
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from peft import LoraConfig


DEFAULT_CHATML_CHAT_TEMPLATE = "{% for message in messages %}\n{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% if loop.last and add_generation_prompt %}{{'<|im_start|>assistant\n' }}{% endif %}{% endfor %}"
DEFAULT_ZEPHYR_CHAT_TEMPLATE = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"


class ZephyrSpecialTokens(str, Enum):
    user = "<|user|>"
    assistant = "<|assistant|>"
    system = "<|system|>"
    eos_token = "</s>"
    bos_token = "<s>"
    pad_token = "<pad>"

    @classmethod
    def list(cls):
        return [c.value for c in cls]


class ChatmlSpecialTokens(str, Enum):
    user = "<|im_start|>user"
    assistant = "<|im_start|>assistant"
    system = "<|im_start|>system"
    eos_token = "<|im_end|>"
    bos_token = "<s>"
    pad_token = "<pad>"

    @classmethod
    def list(cls):
        return [c.value for c in cls]


def create_datasets(tokenizer, data_args, training_args, apply_chat_template=False):
    def preprocess(samples):
        batch = []
        for conversation in samples["messages"]:
            batch.append(tokenizer.apply_chat_template(conversation, tokenize=False))
        return {"content": batch}

    raw_datasets = DatasetDict()
    for split in data_args.splits.split(","):
        try:
            # Try first if dataset on a Hub repo
            dataset = load_dataset(data_args.dataset_name, split=split)
        except DatasetGenerationError:
            # If not, check local dataset
            dataset = load_from_disk(os.path.join(data_args.dataset_name, split))

        if "train" in split:
            raw_datasets["train"] = dataset
        elif "test" in split:
            raw_datasets["test"] = dataset
        else:
            raise ValueError(f"Split type {split} not recognized as one of test or train.")

    if apply_chat_template:
        raw_datasets = raw_datasets.map(
            preprocess,
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
        )

    train_data = raw_datasets["train"]
    valid_data = raw_datasets["test"]
    print(f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}")
    print(f"A sample of train dataset: {train_data[0]}")

    return train_data, valid_data


def create_and_prepare_model(args, data_args, training_args):
    bnb_config = None
    quant_storage_dtype = None

    if args.use_4bit_quantization:
        compute_dtype = getattr(torch, args.bnb_4bit_compute_dtype)
        quant_storage_dtype = getattr(torch, args.bnb_4bit_quant_storage_dtype)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=args.use_4bit_quantization,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=args.use_nested_quant,
            bnb_4bit_quant_storage=quant_storage_dtype,
        )

        if compute_dtype == torch.float16 and args.use_4bit_quantization:
            major, _ = torch.cuda.get_device_capability()
            if major >= 8:
                print("=" * 80)
                print("Your GPU supports bfloat16, you can accelerate training with the argument --bf16")
                print("=" * 80)
        elif args.use_8bit_quantization:
            bnb_config = BitsAndBytesConfig(load_in_8bit=args.use_8bit_quantization)

    torch_dtype = (
        quant_storage_dtype if quant_storage_dtype and quant_storage_dtype.is_floating_point else torch.float32
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        quantization_config=bnb_config,
        trust_remote_code=True,
        attn_implementation="sdpa" if args.use_flash_attn else "eager",
        torch_dtype=torch_dtype,
    )

    peft_config = LoraConfig(
        base_model_name_or_path=args.model_name_or_path,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        r=args.lora_r,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=args.lora_target_modules.split(",")
        if args.lora_target_modules != "all-linear"
        else args.lora_target_modules,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    return model, peft_config, tokenizer