from collections.abc import Iterable, Sequence
from typing import Any


def select_calibration_texts(
    texts: Iterable[str],
    sample_count: int,
) -> list[str]:
    """Select a fixed number of non-empty calibration texts."""

    if (
        not isinstance(sample_count, int)
        or isinstance(sample_count, bool)
        or sample_count <= 0
    ):
        raise ValueError("Sample count must be a positive integer.")

    selected = []

    for text in texts:
        if not isinstance(text, str):
            continue

        cleaned_text = " ".join(text.split())

        if cleaned_text:
            selected.append(cleaned_text)

        if len(selected) == sample_count:
            break

    if len(selected) < sample_count:
        raise ValueError(
            f"Requested {sample_count} calibration samples, "
            f"but found only {len(selected)} non-empty texts."
        )

    return selected


def load_wikitext_calibration_texts(
    sample_count: int,
) -> list[str]:
    """Load calibration texts from the WikiText training split."""

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Install the 'datasets' package to load WikiText."
        ) from exc

    dataset = load_dataset(
        "Salesforce/wikitext",
        "wikitext-2-raw-v1",
        split="train",
        streaming=True,
    )

    texts = (
        record.get("text", "")
        for record in dataset
    )

    return select_calibration_texts(texts, sample_count)


def tokenize_calibration_texts(
    texts: Sequence[str],
    tokenizer: Any,
    max_seq_length: int,
) -> list[dict[str, list[int]]]:
    """Tokenize calibration texts for the quantization workflow."""

    if (
        not isinstance(max_seq_length, int)
        or isinstance(max_seq_length, bool)
        or max_seq_length <= 0
    ):
        raise ValueError("Maximum sequence length must be positive.")

    if not texts:
        raise ValueError("At least one calibration text is required.")

    tokenized_records = []

    for text in texts:
        if not isinstance(text, str) or not text.strip():
            continue

        encoded = tokenizer(
            text,
            padding=False,
            truncation=True,
            max_length=max_seq_length,
            add_special_tokens=False,
        )

        input_ids = encoded.get("input_ids")

        if input_ids is None or len(input_ids) == 0:
            continue

        attention_mask = encoded.get(
            "attention_mask",
            [1] * len(input_ids),
        )

        if len(input_ids) != len(attention_mask):
            raise ValueError(
                "Input IDs and attention mask lengths do not match."
            )

        tokenized_records.append({
            "input_ids": [int(token) for token in input_ids],
            "attention_mask": [int(mask) for mask in attention_mask],
        })

    if not tokenized_records:
        raise ValueError("No valid tokenized calibration examples found.")

    return tokenized_records

def load_wikitext_evaluation_texts(
    sample_count: int,
) -> list[str]:
    """Load held-out WikiText-2 test examples for evaluation."""

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Install the 'datasets' package to load WikiText."
        ) from exc

    dataset = load_dataset(
        "Salesforce/wikitext",
        "wikitext-2-raw-v1",
        split="test",
        streaming=True,
    )

    texts = (
        record.get("text", "")
        for record in dataset
    )

    return select_calibration_texts(texts, sample_count)