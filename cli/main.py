import click
import subprocess
import importlib.resources


def get_argparse_defaults(parser):
    defaults = {}
    for action in parser._actions:
        if not action.required and action.dest != "help":
            defaults[action.dest] = action.default
    return defaults


@click.group()
def main():
    pass



@main.command()
@click.option("--data_train", required=False, type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True), help="Path to the training data.")
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
    accelerate_config_filename = importlib.resources.files("cli.data") / "fsdp_config_qlora.yaml"
    accelerate_train_filename = importlib.resources.files("cli.scripts") / "train.py"
    accelerate_bash_filename = importlib.resources.files("cli.data") / "launch.sh"
    proc = subprocess.Popen(['bash', accelerate_bash_filename], env={
        'FSDP_CONFIG': accelerate_config_filename,
        'TRAIN_SCRIPT': accelerate_train_filename,
        'TRAIN_FILENAME': '/clips-ai-train/datasets/system_prompt/train.jsonl',
        'HF_TOKEN': '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'
    })
    proc.wait()

    #accelerate_args, *_ = _validate_launch_command(accelerate_args)
    #multi_gpu_launcher(accelerate_args)
    # Initiate a training argument manually and pass it to the fine-tuning function;
    click.echo('Start training')


@main.command()
def generate():
    click.echo('Start generation')