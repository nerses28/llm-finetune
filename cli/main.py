import click
from argparse import Namespace
from accelerate.commands.launch import multi_gpu_launcher, _validate_launch_command, launch_command_parser


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
    "lr_schedule_type": "constant",
    "weight_decay": "1e-4",
    "warmup_ratio": 0.1,
    "max_grad_norm": 1.0
}



@main.command()
@click.option("--data_train", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True), help="Path to the training data.")
@click.option("--data_valid", type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True), help="Path to the validation data.")
@click.option("--max_seq_len", default=2048, type=click.IntRange(min=8, max=8192), show_default=True, help="Maximum sequence length of the model.")
@click.option("--lora_r", default=32, type=click.IntRange(min=8, max=256), show_default=True, help="Size of the LoRA adapters (i.e., r from the paper)")
@click.option("--lora_alpha", default=64, type=click.IntRange(min=8, max=512), show_default=True, help="Scaling factor of the LoRA adapters (i.e., alpha from the paper)")
@click.option("--lora_dropout", default=0.0, show_default=True, type=click.FloatRange(min=0.0, max=0.5), help="Dropout applied to the LoRA adapters (i.e., r from the paper)")
@click.option("--batch_size", default=32, show_default=True, type=click.IntRange(min=1, max=512))
@click.option("--learning_rate", default=0.00005, show_default=True, type=click.FloatRange(min=0.0, max=0.1))
@click.option("--n_epoch", default=1, show_default=True, type=click.FloatRange(min=0.001, max=10**9))
@click.option("--model_suffix", type=str, help="Suffix applied to the adapter name")
def train(*args, **kwargs):
    # Trainer arguments
    trainer_args = {
        **DEFAULT_ARGUMENTS,
        "train_filename": 
    }
    # Accelerate arguments
    parser = launch_command_parser()
    accelerate_args = get_argparse_defaults(parser)
    accelerate_args["config_file"] = "/clips-ai-train/fsdp_config_qlora.yaml"
    accelerate_args["training_script"] = "/root/train.py"
    # have the default arguments stored somewhere
    accelerate_args["training_script_args"] = ""
    accelerate_args = Namespace(**accelerate_args)
    accelerate_args, *_ = _validate_launch_command(accelerate_args)
    multi_gpu_launcher(accelerate_args)
    # Initiate a training argument manually and pass it to the fine-tuning function;
    click.echo('Start training')


@main.command()
def generate():
    click.echo('Start generation')