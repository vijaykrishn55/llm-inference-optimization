import math
from collections.abc import Sequence
from typing import Any

from src.inference_pipeline.evaluation_metrics import (
    calculate_accuracy_degradation,
    calculate_perplexity_change,
    perplexity_from_loss,
)


def validate_evaluation_inputs(
    texts: Sequence[str],
    max_seq_length: int,
) -> list[str]:
    """Validate and clean text examples for evaluation."""

    if max_seq_length < 2:
        raise ValueError("Maximum sequence length must be at least 2.")

    if not texts:
        raise ValueError("At least one evaluation text is required.")

    cleaned_texts = [
        text.strip()
        for text in texts
        if isinstance(text, str) and text.strip()
    ]

    if not cleaned_texts:
        raise ValueError("No valid evaluation texts were provided.")

    return cleaned_texts


def calculate_perplexity_from_nll(
    total_negative_log_likelihood: float,
    token_count: int,
) -> dict:
    """Calculate mean loss and perplexity from total token loss."""

    if (
        not math.isfinite(total_negative_log_likelihood)
        or total_negative_log_likelihood < 0
    ):
        raise ValueError("Total negative log-likelihood must be finite and non-negative.")

    if (
        not isinstance(token_count, int)
        or isinstance(token_count, bool)
        or token_count <= 0
    ):
        raise ValueError("Token count must be a positive integer.")

    mean_loss = total_negative_log_likelihood / token_count

    return {
        "mean_loss": mean_loss,
        "perplexity": perplexity_from_loss(mean_loss),
        "tokens_evaluated": token_count,
    }


def evaluate_model_perplexity(
    model: Any,
    tokenizer: Any,
    texts: Sequence[str],
    max_seq_length: int = 512,
) -> dict:
    """Evaluate causal-LM perplexity and next-token accuracy.

    This function requires PyTorch and a model that returns loss and logits.
    It runs the evaluation examples individually to avoid padding.
    """

    import torch

    valid_texts = validate_evaluation_inputs(
        texts,
        max_seq_length,
    )

    model.eval()
    device = next(model.parameters()).device

    total_negative_log_likelihood = 0.0
    total_tokens = 0
    total_correct = 0
    evaluated_examples = 0

    with torch.inference_mode():
        for text in valid_texts:
            encoded = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_seq_length,
                add_special_tokens=True,
            )

            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)

            sequence_length = input_ids.shape[1]

            if sequence_length < 2:
                continue

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
            )

            target_tokens = sequence_length - 1

            total_negative_log_likelihood += (
                float(outputs.loss.item()) * target_tokens
            )

            predictions = outputs.logits[:, :-1, :].argmax(dim=-1)
            expected_tokens = input_ids[:, 1:]

            total_correct += int(
                (predictions == expected_tokens).sum().item()
            )

            total_tokens += target_tokens
            evaluated_examples += 1

    if total_tokens == 0:
        raise ValueError(
            "No examples contained enough tokens for evaluation."
        )

    result = calculate_perplexity_from_nll(
        total_negative_log_likelihood,
        total_tokens,
    )

    result["accuracy"] = total_correct / total_tokens
    result["examples_evaluated"] = evaluated_examples

    return result


def build_evaluation_report(
    baseline: dict,
    quantized: dict,
) -> dict:
    """Compare baseline and quantized model evaluation results."""

    perplexity_comparison = calculate_perplexity_change(
        baseline["perplexity"],
        quantized["perplexity"],
    )

    accuracy_comparison = calculate_accuracy_degradation(
        baseline["accuracy"],
        quantized["accuracy"],
    )

    return {
        "baseline": baseline,
        "quantized": quantized,
        "perplexity_comparison": perplexity_comparison,
        "accuracy_comparison": accuracy_comparison,
    }