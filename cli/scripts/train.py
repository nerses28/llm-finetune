# Adapted from https://github.com/huggingface/peft/blob/main/examples/sft/train.py

import os
import sys
import json
from dataclasses import dataclass, field
from typing import Optional


# Define and parse arguments.
@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    lora_alpha: Optional[int] = field(default=16)
    lora_dropout: Optional[float] = field(default=0.1)
    lora_r: Optional[int] = field(default=64)


@dataclass
class DataTrainingArguments:
    train_filename: str = field(metadata={"help": "Path to the training data."})
    valid_filename: Optional[str] = field(default=None, metadata={"help": "Path to the validation data."})
    max_seq_length: Optional[int] = field(default=512)


def main(model_args, data_args, training_args):
    import torch

    from transformers import HfArgumentParser, TrainingArguments, set_seed
    from trl import SFTTrainer
    from datasets import load_dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
    from peft import LoraConfig


    def get_datasets(data_args):
        data_files = {"train": data_args.train_filename}
        if data_args.valid_filename is not None:
            data_files["validation"] = data_args.valid_filename
        raw_dataset = load_dataset("json", data_files=data_files)
        return raw_dataset["train"], raw_dataset.get("validation", None)


    def get_model(args, data_args, training_args):
        # Quantization params
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_storage=torch.uint8,
        )

        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            quantization_config=bnb_config,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True
        )

        peft_config = LoraConfig(
            base_model_name_or_path=args.model_name_or_path,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            r=args.lora_r,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules="all-linear"
        )

        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        tokenizer.pad_token = tokenizer.eos_token
        return model, peft_config, tokenizer

    # Set seed for reproducibility
    set_seed(training_args.seed)

    # model
    model, peft_config, tokenizer = get_model(model_args, data_args, training_args)

    # gradient ckpt
    assert training_args.gradient_checkpointing
    model.config.use_cache = not training_args.gradient_checkpointing
    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}

    # datasets
    train_dataset, eval_dataset = get_datasets(data_args)

    # trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        packing=True,
        dataset_kwargs={
            "append_concat_token": True,
        },
        max_seq_length=data_args.max_seq_length,
    )
    trainer.accelerator.print(f"{trainer.model}")
    trainer.model.print_trainable_parameters()

    # train
    trainer.train()

    # saving final model
    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    trainer.save_model()


if __name__ == "__main__":
    print(sys.argv)
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    main(model_args, data_args, training_args)