import click
import importlib.resources
from argparse import Namespace
from transformers import TrainingArguments
from cli.scripts.train import main as train_fn
# from accelerate.commands.launch import multi_gpu_launcher, _validate_launch_command, launch_command_parser, launch_command
from accelerate import notebook_launcher
from dataclasses import dataclass, field
from typing import Optional


def get_argparse_defaults(parser):
    defaults = {}
    for action in parser._actions:
        if not action.required and action.dest != "help":
            defaults[action.dest] = action.default
    return defaults


@click.group()
def main():
    pass


@dataclass
class LocalTrainingArguments(TrainingArguments):
    def __post_init__(self):
        pass


@dataclass
class DataTrainingArguments:
    train_filename: str = field(metadata={"help": "Path to the training data."})
    valid_filename: Optional[str] = field(default=None, metadata={"help": "Path to the validation data."})
    max_seq_length: Optional[int] = field(default=512)


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



@main.command()
@click.option("--data_train", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True), help="Path to the training data.")
@click.option("--data_valid", type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True), help="Path to the validation data.")
@click.option("--max_seq_len", default=1024, type=click.IntRange(min=8, max=8192), show_default=True, help="Maximum sequence length of the model.")
@click.option("--lora_r", default=32, type=click.IntRange(min=8, max=256), show_default=True, help="Size of the LoRA adapters (i.e., r from the paper)")
@click.option("--lora_alpha", default=64, type=click.IntRange(min=8, max=512), show_default=True, help="Scaling factor of the LoRA adapters (i.e., alpha from the paper)")
@click.option("--lora_dropout", default=0.0, show_default=True, type=click.FloatRange(min=0.0, max=0.5), help="Dropout applied to the LoRA adapters (i.e., r from the paper)")
@click.option("--batch_size", default=4, show_default=True, type=click.IntRange(min=1, max=512))
@click.option("--learning_rate", default=0.00005, show_default=True, type=click.FloatRange(min=0.0, max=0.1))
@click.option("--n_epoch", default=1, show_default=True, type=click.FloatRange(min=0.001, max=10**9))
@click.option("--model_suffix", type=str, help="Suffix applied to the adapter name")
def train(data_train, data_valid, max_seq_len, lora_r, lora_alpha, lora_dropout, batch_size, learning_rate, n_epoch, model_suffix):
    # Trainer arguments
    training_args = LocalTrainingArguments(
        output_dir="test_output_dir",
        num_train_epochs=n_epoch,
        seed=100,
        logging_steps=50,
        log_level="info",
        logging_strategy="steps",
        eval_strategy="epoch",
        save_strategy="no",
        bf16=True,
        lr_scheduler_type="constant",
        weight_decay="1e-4",
        warmup_ratio=0.1,
        max_grad_norm=1.0,
        gradient_checkpointing=True,
        remove_unused_columns=True
    )
    model_args = ModelArguments(
        model_name_or_path="meta-llama/Meta-Llama-3-8B-Instruct",
        lora_alpha=lora_alpha,
        lora_r=lora_r,
        lora_dropout=lora_dropout
    )
    data_args = DataTrainingArguments(
        train_filename=data_train,
        valid_filename=data_valid,
        max_seq_length=max_seq_len
    )
    notebook_launcher(train_fn, (model_args, data_args, training_args), num_processes=2)
    #accelerate_args, *_ = _validate_launch_command(accelerate_args)
    #multi_gpu_launcher(accelerate_args)
    # Initiate a training argument manually and pass it to the fine-tuning function;
    click.echo('Start training')


@main.command()
def generate():
    click.echo('Start generation')