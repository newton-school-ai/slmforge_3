"""src/slmforge/finetune/train.py.
=============================
CLI entrypoint for LoRA fine-tuning.
"""

from __future__ import annotations

import typer

from slmforge.finetune.lora import train_lora

app = typer.Typer(
    help="SLMForge fine-tuning CLI -- train LoRA adapters on base models.",
)


@app.command()
def main(
    dataset: str = typer.Option(
        ...,
        "--dataset",
        "-d",
        help="Path or HF ID of dataset to train on.",
    ),
    base: str = typer.Option(
        "phi-3-mini",
        "--base",
        "-b",
        help="Base model name from registry or HF repo ID.",
    ),
    output_dir: str = typer.Option(
        "./outputs/adapter",
        "--output-dir",
        "-o",
        help="Directory to save the trained adapter.",
    ),
    epochs: float = typer.Option(
        1.0,
        "--epochs",
        "-e",
        help="Number of training epochs.",
    ),
    learning_rate: float = typer.Option(
        2e-4,
        "--lr",
        help="Learning rate.",
    ),
    r: int = typer.Option(
        8,
        "--lora-r",
        help="LoRA rank parameter.",
    ),
    alpha: int = typer.Option(
        16,
        "--lora-alpha",
        help="LoRA alpha scaling parameter.",
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        "-s",
        help="Random seed for reproducibility.",
    ),
) -> None:
    """Train a LoRA adapter using dataset and base model."""
    typer.echo(f"Starting LoRA training with base model: {base}")

    lora_cfg = {"r": r, "lora_alpha": alpha}
    training_cfg = {
        "output_dir": output_dir,
        "num_train_epochs": epochs,
        "learning_rate": learning_rate,
        "seed": seed,
    }

    adapter_path = train_lora(
        dataset=dataset,
        base=base,
        lora_cfg=lora_cfg,
        training_cfg=training_cfg,
    )

    typer.echo(f"Training complete! Adapter saved to: {adapter_path}")


if __name__ == "__main__":
    app()
