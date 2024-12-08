import os

import sys
import json
from typing import Optional
from dataclasses import dataclass, field
from accelerate import Accelerator

from transformers import HfArgumentParser, set_seed

import torch

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from peft import LoraConfig, PeftModel

import logging
import warnings

# Set the global logging level to ERROR to suppress warnings
logger = logging.getLogger("transformers.tokenization_utils_base")
logger.setLevel(logging.ERROR)
warnings.filterwarnings(
    "ignore",
    message="Upcasted low precision parameters in Linear because mixed precision turned on in FSDP.*",
)
warnings.filterwarnings(
    "ignore",
    message="FSDP upcast of low precision parameters may affect the precision of model checkpoints.*",
)

@dataclass
class PredictArguments:
    """
    Arguments for controlling the prediction generation.
    """
    max_new_tokens: int = field(
        default=128,
        metadata={"help": "The maximum number of new tokens to generate."}
    )
    num_return_sequences: int = field(
        default=1,
        metadata={"help": "The number of sequences to generate for each input."}
    )
    temperature: float = field(
        default=0.7,
        metadata={"help": "Sampling temperature. Lower values make the model more focused, higher values more creative."}
    )
    top_p: float = field(
        default=0.9,
        metadata={"help": "Nucleus sampling probability. Lower values make the model choose from high-probability tokens only."}
    )
    batch_size: int = field(
        default=1,
        metadata={"help": "The number of examples to process in a single batch during prediction."}
    )
    
@dataclass
class ModelArguments:
    peft_path: str = field(
        metadata={"help": "Path to the trained LoRA adapter."}
    )

@dataclass
class DataArguments:
    input_filename: str = field(metadata={"help": "Path to the input data."})
    output_filename: str = field(metadata={"help": "File to save the predictions in JSONL format."})
    output_path: str = field(metadata={"help": "Path to save the predictions in JSONL format."})

def get_datasets(data_args):
    raw_dataset = load_dataset("json", data_files={"input": data_args.input_filename})
    return raw_dataset["input"]

def apply_chat_template(dataset, tokenizer):
    if dataset is None:
        return None

    def preprocess(example):
        return {
            "text": tokenizer.apply_chat_template(example["messages"][:-1], tokenize=False)
        }

    return dataset.map(preprocess)

def load_model_and_tokenizer(model_args):
    """
    Load the model and tokenizer with the specified adapter and base model configuration.
    """
    # Load adapter config to determine base model path if not provided
    peft_config = LoraConfig.from_pretrained(model_args.peft_path)
    base_model_path = peft_config.base_model_name_or_path

    # Set up 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_storage=torch.bfloat16,
    )

    # Load base model and adapter
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        quantization_config=bnb_config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    )
    model = PeftModel.from_pretrained(model, model_args.peft_path)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.peft_path,
        trust_remote_code=True,
    )
    tokenizer.padding_side = "left"
    desired_pad_token_id = 128001
    tokenizer.pad_token_id = desired_pad_token_id
    model.config.pad_token_id = desired_pad_token_id
    
    return model, tokenizer

def main(model_args, data_args, predict_args):
    # Set up Accelerator
    accelerator = Accelerator()
    accelerator.print("Loading model and tokenizer...")
    
    model, tokenizer = load_model_and_tokenizer(model_args)
    model = model.to(accelerator.device)
    #model = accelerator.prepare(model)
        
    # Set seed for reproducibility
    set_seed(42)

    # Load and preprocess dataset
    accelerator.print("Loading and preprocessing dataset...")
    
    with accelerator.main_process_first():
        dataset = get_datasets(data_args)
        dataset = apply_chat_template(dataset, tokenizer)
    # Prepare for generation
    accelerator.print("Generating predictions...")
    model.eval()
    
    results = []
    batch_size = predict_args.batch_size
    num_batches = (len(dataset) + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(dataset))
        print(f"Processing batch {batch_idx + 1}/{num_batches} ({start_idx + 1}-{end_idx})", end='\r')

        batch = dataset[start_idx:end_idx]
        inputs = tokenizer(
            batch["text"], 
            return_tensors="pt", 
            padding=True, 
            truncation=True
        )
        inputs = inputs.to(accelerator.device)
        #inputs = accelerator.prepare(inputs)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=predict_args.max_new_tokens,
                num_return_sequences=predict_args.num_return_sequences,
                temperature=predict_args.temperature,
                pad_token_id=model.config.pad_token_id,
                top_p=predict_args.top_p
            )

        predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for text, prediction in zip(batch["text"], predictions):
                results.append({
                    "input": text,
                    "prediction": prediction,
                })

    print("\nPrediction generation complete.")

    # Save predictions in JSONL format
    accelerator.print(f"Saving predictions to {data_args.output_filename}...")
    os.makedirs(os.path.dirname(data_args.output_path), exist_ok=True)
    with open(data_args.output_path + data_args.output_filename, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, DataArguments, PredictArguments))
    model_args, data_args, predict_args = parser.parse_args_into_dataclasses()
    main(model_args, data_args, predict_args)
