import math

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def calculate_perplexity(
    model,
    tokenizer,
    texts: list[str],
    max_length: int = 512,
) -> float:

    model.eval()

    total_loss = 0.0
    total_tokens = 0

    device = next(model.parameters()).device

    with torch.no_grad():

        for text in texts:

            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            )

            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
            )

            token_count = attention_mask.sum().item()

            total_loss += outputs.loss.item() * token_count
            total_tokens += token_count

    average_loss = total_loss / total_tokens

    return math.exp(average_loss)


def load_wikitext_samples(
    split: str = "test",
    max_samples: int = 100,
) -> list[str]:

    dataset = load_dataset(
        "wikitext",
        "wikitext-2-raw-v1",
        split=split,
    )

    texts = [
        item["text"]
        for item in dataset
        if item["text"].strip()
    ]

    return texts[:max_samples]