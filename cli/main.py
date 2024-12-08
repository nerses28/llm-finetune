import uuid
import click
import getpass
import importlib.resources

from accelerate.commands.launch import launch_command
from cli.utils import accelerate_args, training_args


DEFAULT_SAVING_DIR = "output"


def get_prefix(prefix=""):
    user = getpass.getuser()
    if prefix.strip() != "":
        return f"{user}_{prefix}"
    return user


@click.group()
def main():
    pass


@main.command()
@click.option(
    "--data_train",
    required=False,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
    help="Path to the training data.",
)
@click.option(
    "--data_valid",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
    help="Path to the validation data.",
)
@click.option(
    "--max_seq_len",
    default=1024,
    type=click.IntRange(min=8, max=8192),
    show_default=True,
    help="Maximum sequence length of the model.",
)
@click.option(
    "--lora_r",
    default=32,
    type=click.IntRange(min=8, max=256),
    show_default=True,
    help="Size of the LoRA adapters (i.e., r from the paper)",
)
@click.option(
    "--lora_alpha",
    default=64,
    type=click.IntRange(min=8, max=512),
    show_default=True,
    help="Scaling factor of the LoRA adapters (i.e., alpha from the paper)",
)
@click.option(
    "--lora_dropout",
    default=0.0,
    show_default=True,
    type=click.FloatRange(min=0.0, max=0.5),
    help="Dropout applied to the LoRA adapters (i.e., r from the paper)",
)
@click.option(
    "--batch_size", default=4, show_default=True, type=click.IntRange(min=1, max=512)
)
@click.option(
    "--learning_rate",
    default=0.00005,
    show_default=True,
    type=click.FloatRange(min=0.0, max=0.1),
)
@click.option(
    "--n_epoch",
    default=1,
    show_default=True,
    type=click.FloatRange(min=0.001, max=10**9),
)
@click.option(
    "--wandb/--no-wandb",
    default=False
)
@click.option("--model_suffix", type=str, default="", help="Suffix applied to the adapter name")
def train(
    data_train,
    data_valid,
    max_seq_len,
    lora_r,
    lora_alpha,
    lora_dropout,
    batch_size,
    learning_rate,
    n_epoch,
    model_suffix,
    wandb
):
    accelerate_train_filename = importlib.resources.files("cli.scripts") / "train.py"
    accelerate_config_filename = (
        importlib.resources.files("cli.data") / "fsdp_config_qlora.yaml"
    )
    output_dir = DEFAULT_SAVING_DIR + "/" + get_prefix(prefix="train") + "_" + str(uuid.uuid4())[:6] + "_" + model_suffix
    print(output_dir)
    training_script_args = {
        "train_filename": str(data_train),
        "valid_filename": str(data_valid),
        "max_seq_len": str(max_seq_len),
        "num_train_epoch": str(n_epoch),
        "learning_rate": str(learning_rate),
        "per_device_train_batch_size": str(batch_size),
        "per_device_eval_batch_size": str(batch_size),
        "lora_r": str(lora_r),
        "lora_alpha": str(lora_alpha),
        "lora_dropout": str(lora_dropout),
        "packing": "True",
        "dataset_text_field": "text",
        "report_to": "wandb" if wandb else "none",
        "output_dir": output_dir,
        "lora_target_modules": "all-linear",
        **training_args,
    }
    training_script_args = " ".join(
        [f"--{k} {v}" for k, v in training_script_args.items() if v is not None]
    )
    training_script_args = training_script_args.split()
    accelerate_args.config_file = str(accelerate_config_filename)
    accelerate_args.training_script = str(accelerate_train_filename)
    accelerate_args.training_script_args = training_script_args
    launch_command(accelerate_args)

@main.command()
@click.option(
    "--data_test",
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
    help="Path to the test data for prediction.",
)
@click.option(
    "--peft_path",
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=True, readable=True),
    help="Path to the trained LoRA adapter.",
)
@click.option(
    "--output_filename",
    default="predictions.jsonl",
    show_default=True,
    type=click.Path(file_okay=True, dir_okay=False, writable=True),
    help="File where predictions will be saved (JSONL format).",
)
@click.option(
    "--output_path",
    default="predict_results/",
    show_default=True,
    type=click.Path(file_okay=False, dir_okay=True, writable=True),
    help="Path where predictions will be saved.",
)
@click.option(
    "--max_new_tokens",
    default=128,
    show_default=True,
    type=click.IntRange(min=1, max=1024),
    help="Maximum number of tokens to generate.",
)
@click.option(
    "--temperature",
    default=0.7,
    show_default=True,
    type=click.FloatRange(min=0.0, max=2.0),
    help="Sampling temperature for generation.",
)
@click.option(
    "--top_p",
    default=0.9,
    show_default=True,
    type=click.FloatRange(min=0.0, max=1.0),
    help="Nucleus sampling probability for generation.",
)
@click.option(
    "--num_return_sequences",
    default=1,
    show_default=True,
    type=click.IntRange(min=1, max=10),
    help="Number of sequences to return for each input.",
)
@click.option(
    "--batch_size", default=1, show_default=True, type=click.IntRange(min=1, max=512)
)
def predict(
    data_test,
    peft_path,
    output_filename,
    output_path,
    max_new_tokens,
    temperature,
    top_p,
    num_return_sequences,
    batch_size,
):
    accelerate_predict_filename = importlib.resources.files("cli.scripts") / "predict.py"
    accelerate_config_filename = (
        importlib.resources.files("cli.data") / "fsdp_config_qlora.yaml"
    )

    prediction_script_args = {
        "input_filename": str(data_test),
        "peft_path": str(peft_path),
        "output_filename": str(output_filename),
        "output_path": str(output_path),
        "max_new_tokens": str(max_new_tokens),
        "temperature": str(temperature),
        "top_p": str(top_p),
        "num_return_sequences": str(num_return_sequences),
        "batch_size": str(batch_size),
    }
    prediction_script_args = " ".join(
        [f"--{k} {v}" for k, v in prediction_script_args.items() if v is not None]
    )
    prediction_script_args = prediction_script_args.split()
    accelerate_args.config_file = str(accelerate_config_filename)
    accelerate_args.training_script = str(accelerate_predict_filename)
    accelerate_args.training_script_args = prediction_script_args
    launch_command(accelerate_args)

    
@main.command()
@click.option(
    "--data_train",
    required=False,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
    help="Path to the training data.",
)
@click.option(
    "--data_valid",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
    help="Path to the validation data.",
)
@click.option(
    "--max_seq_len",
    default=1024,
    type=click.IntRange(min=8, max=8192),
    show_default=True,
    help="Maximum sequence length of the model.",
)
@click.option(
    "--batch_size", default=4, show_default=True, type=click.IntRange(min=1, max=512)
)
@click.option(
    "--learning_rate",
    default=0.00005,
    show_default=True,
    type=click.FloatRange(min=0.0, max=0.1),
)
@click.option(
    "--n_epoch",
    default=1,
    show_default=True,
    type=click.FloatRange(min=0.001, max=10**9),
)
@click.option("--model_suffix", type=str, help="Suffix applied to the adapter name")
@click.option("--peft_path", type=str, help="Path to the SFT adapter")
def align(
    data_train,
    data_valid,
    max_seq_len,
    batch_size,
    learning_rate,
    n_epoch,
    model_suffix,
    peft_path,
):
    accelerate_train_filename = importlib.resources.files("cli.scripts") / "align.py"
    accelerate_config_filename = (
        importlib.resources.files("cli.data") / "fsdp_config_qlora.yaml"
    )
    output_dir = DEFAULT_SAVING_DIR + "/" + get_prefix(prefix="align") + "_" + str(uuid.uuid4())[:6] + "_" + model_suffix
    training_script_args = {
        "train_filename": str(data_train),
        "valid_filename": str(data_valid),
        "num_train_epoch": str(n_epoch),
        "learning_rate": str(learning_rate),
        "per_device_train_batch_size": str(batch_size),
        "per_device_eval_batch_size": str(batch_size),
        "peft_path": str(peft_path),
        "model_adapter_name": "aligned",
        "ref_adapter_name": "reference",
        "max_length": max_seq_len,
        "max_prompt_length": max_seq_len,
        "remove_unused_columns": "False",
        "output_dir": output_dir,
        **training_args,
    }
    training_script_args = " ".join(
        [f"--{k} {v}" for k, v in training_script_args.items() if v is not None]
    )
    training_script_args = training_script_args.split()
    accelerate_args.config_file = str(accelerate_config_filename)
    accelerate_args.training_script = str(accelerate_train_filename)
    accelerate_args.training_script_args = training_script_args
    launch_command(accelerate_args)


@main.command()
def generate():
    click.echo("Start generation")
