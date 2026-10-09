import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import load_config
from src.evaluation.perplexity import (
    calculate_perplexity,
    load_wikitext_samples,
)


def evaluate_model(model_path: str) -> float:

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    texts = load_wikitext_samples(
        max_samples=100,
    )

    return calculate_perplexity(
        model,
        tokenizer,
        texts,
        max_length=512,
    )


def main():

    config = load_config()

    model_id = config["model"]["id"]
    quantized_model = config["model"]["quantized_dir"]

    print("Evaluating FP16 model...")
    fp16_ppl = evaluate_model(model_id)

    print("Evaluating quantized model...")
    quantized_ppl = evaluate_model(quantized_model)

    degradation = (
        (quantized_ppl - fp16_ppl)
        / fp16_ppl
    ) * 100

    result = {
        "fp16_perplexity": fp16_ppl,
        "quantized_perplexity": quantized_ppl,
        "degradation_percent": degradation,
    }

    Path("results").mkdir(exist_ok=True)

    with open(
        "results/perplexity.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(result, file, indent=2)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()