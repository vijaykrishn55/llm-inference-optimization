from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform.awq import AWQModifier


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load YAML configuration."""
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def quantize_model(config_path: str = "configs/config.yaml") -> None:
    """Quantize the FP16 model using AWQ 4-bit quantization."""

    config = load_config(config_path)

    model_id = config["model"]["id"]
    output_dir = Path(config["model"]["output_dir"])

    num_samples = config["quantization"]["num_calibration_samples"]
    max_seq_length = config["quantization"]["max_seq_length"]

    dataset_id = config["calibration"]["dataset"]
    dataset_split = config["calibration"]["split"]

    print(f"Loading model: {model_id}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float16,
        device_map="auto",
    )

    print("Loading calibration dataset...")

    dataset = load_dataset(
        dataset_id,
        split=f"{dataset_split}[:{num_samples}]",
    )

    dataset = dataset.shuffle(seed=42)

    def preprocess(example):
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
        )

        return tokenizer(
            text,
            padding=False,
            truncation=True,
            max_length=max_seq_length,
            add_special_tokens=False,
        )

    dataset = dataset.map(
        preprocess,
        remove_columns=dataset.column_names,
    )

    print(f"Calibration samples: {len(dataset)}")

    # AWQ + 4-bit weight quantization.
    recipe = [
        AWQModifier(),
        QuantizationModifier(
            ignore=["lm_head"],
            scheme="W4A16_ASYM",
            targets=["Linear"],
        ),
    ]

    print("Starting AWQ quantization...")

    oneshot(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        recipe=recipe,
        max_seq_length=max_seq_length,
        num_calibration_samples=num_samples,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    print("Saving quantized model...")

    model.save_pretrained(
        output_dir,
        save_compressed=True,
    )

    tokenizer.save_pretrained(output_dir)

    print(f"\nQuantized model saved to: {output_dir}")


if __name__ == "__main__":
    quantize_model()