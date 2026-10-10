import sys
import types

import pytest

from src.inference_pipeline.calibration_data import (
    select_calibration_texts,
    load_wikitext_calibration_texts,
    tokenize_calibration_texts,
)


class FakeTokenizer:
    """Small tokenizer substitute for CPU-only unit tests."""

    def __call__(
        self,
        text,
        padding,
        truncation,
        max_length,
        add_special_tokens,
    ):
        tokens = [ord(char) for char in text][:max_length]

        return {
            "input_ids": tokens,
            "attention_mask": [1] * len(tokens),
        }


def test_select_calibration_texts():
    texts = [" First example ", "", "   ", "Second example", "Third"]

    result = select_calibration_texts(texts, sample_count=2)

    assert result == ["First example", "Second example"]


def test_sample_count_must_be_positive():
    with pytest.raises(ValueError):
        select_calibration_texts(["Example"], sample_count=0)


def test_insufficient_calibration_texts():
    with pytest.raises(ValueError):
        select_calibration_texts(
            ["Only example", " "],
            sample_count=2,
        )


def test_tokenize_calibration_texts():
    tokenizer = FakeTokenizer()

    result = tokenize_calibration_texts(
        ["Hello"],
        tokenizer,
        max_seq_length=3,
    )

    assert result == [
        {
            "input_ids": [ord("H"), ord("e"), ord("l")],
            "attention_mask": [1, 1, 1],
        }
    ]


def test_tokenization_rejects_empty_input():
    with pytest.raises(ValueError):
        tokenize_calibration_texts(
            [],
            FakeTokenizer(),
            max_seq_length=128,
        )


def test_max_sequence_length_must_be_positive():
    with pytest.raises(ValueError):
        tokenize_calibration_texts(
            ["Example"],
            FakeTokenizer(),
            max_seq_length=0,
        )


def test_tokenizer_output_lengths_must_match():
    class InvalidTokenizer:
        def __call__(self, *args, **kwargs):
            return {
                "input_ids": [1, 2],
                "attention_mask": [1],
            }

    with pytest.raises(ValueError):
        tokenize_calibration_texts(
            ["Example"],
            InvalidTokenizer(),
            max_seq_length=128,
        )


def test_wikitext_loader_uses_streaming(monkeypatch):
    fake_datasets = types.ModuleType("datasets")

    def fake_load_dataset(*args, **kwargs):
        assert args == (
            "Salesforce/wikitext",
            "wikitext-2-raw-v1",
        )
        assert kwargs["streaming"] is True

        return iter([
            {"text": ""},
            {"text": " First calibration sample "},
            {"text": "Second calibration sample"},
        ])

    fake_datasets.load_dataset = fake_load_dataset

    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    result = load_wikitext_calibration_texts(sample_count=2)

    assert result == [
        "First calibration sample",
        "Second calibration sample",
    ]

def test_wikitext_evaluation_loader_uses_test_split(monkeypatch):
    import sys
    import types

    fake_datasets = types.ModuleType("datasets")

    def fake_load_dataset(*args, **kwargs):
        assert kwargs["split"] == "test"
        assert kwargs["streaming"] is True

        return iter([
            {"text": ""},
            {"text": "First test example"},
            {"text": "Second test example"},
        ])

    fake_datasets.load_dataset = fake_load_dataset
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    from src.inference_pipeline.calibration_data import (
        load_wikitext_evaluation_texts,
    )

    result = load_wikitext_evaluation_texts(sample_count=2)

    assert result == ["First test example", "Second test example"]