import click
import importlib.resources
from argparse import Namespace
from accelerate.commands.launch import multi_gpu_launcher, _validate_launch_command, launch_command_parser, launch_command


def get_argparse_defaults(parser):
    defaults = {}
    for action in parser._actions:
        if not action.required and action.dest != "help":
            defaults[action.dest] = action.default
    return defaults


@click.group()
def main():
    pass


DEFAULT_ARGUMENTS = {
    "seed": 100,
    "logging_steps": 50,
    "log_level": "info",
    "logging_strategy": "steps",
    "eval_strategy": "epoch",
    "save_strategy": "no",
    "bf16": True,
    "lr_scheduler_type": "constant",
    "weight_decay": "1e-4",
    "warmup_ratio": 0.1,
    "max_grad_norm": 1.0
}



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
    trainer_args = {
        **DEFAULT_ARGUMENTS,
        "model_name_or_path": "meta-llama/Meta-Llama-3-8B-Instruct",
        "num_train_epochs": n_epoch,
        "output_dir": "clips-ai-train/tmp",
        "train_filename": data_train,
        "max_seq_len": max_seq_len,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "gradient_checkpointing": True
    }
    trainer_args = " ".join(["--{k} {v}".format(k=k, v=v) for k, v in trainer_args.items()]).split()
    # Accelerate arguments
    parser = launch_command_parser()
    accelerate_args = get_argparse_defaults(parser)
    accelerate_config_filename = importlib.resources.files("cli.data") / "fsdp_config_qlora.yaml"
    accelerate_train_filename = importlib.resources.files("cli.scripts") / "train.py"
    accelerate_args["config_file"] = str(accelerate_config_filename)
    accelerate_args["training_script"] = str(accelerate_train_filename)
    accelerate_args["training_script_args"] = trainer_args
    accelerate_args = Namespace(**accelerate_args)
    launch_command(accelerate_args)
    #accelerate_args, *_ = _validate_launch_command(accelerate_args)
    #multi_gpu_launcher(accelerate_args)
    # Initiate a training argument manually and pass it to the fine-tuning function;
    click.echo('Start training')


@main.command()
def generate():
    click.echo('Start generation')