import math

import pytest

from src.inference_pipeline.evaluation import (
    validate_evaluation_inputs,
    calculate_perplexity_from_nll,
    build_evaluation_report,
)


def test_validate_evaluation_inputs():
    result = validate_evaluation_inputs(
        [" First example ", "", "Second example"],
        max_seq_length=128,
    )

    assert result == ["First example", "Second example"]


def test_evaluation_rejects_empty_texts():
    with pytest.raises(ValueError):
        validate_evaluation_inputs([" ", ""], max_seq_length=128)


def test_evaluation_requires_at_least_two_tokens():
    with pytest.raises(ValueError):
        validate_evaluation_inputs(["Example"], max_seq_length=1)


def test_calculate_perplexity_from_nll():
    result = calculate_perplexity_from_nll(
        total_negative_log_likelihood=20.0,
        token_count=10,
    )

    assert result["mean_loss"] == pytest.approx(2.0)
    assert result["perplexity"] == pytest.approx(math.exp(2.0))
    assert result["tokens_evaluated"] == 10


def test_perplexity_rejects_negative_nll():
    with pytest.raises(ValueError):
        calculate_perplexity_from_nll(-1.0, 10)


def test_perplexity_rejects_zero_tokens():
    with pytest.raises(ValueError):
        calculate_perplexity_from_nll(10.0, 0)


def test_build_evaluation_report():
    baseline = {
        "mean_loss": 2.0,
        "perplexity": 10.0,
        "accuracy": 0.25,
        "tokens_evaluated": 100,
        "examples_evaluated": 5,
    }

    quantized = {
        "mean_loss": 2.2,
        "perplexity": 12.0,
        "accuracy": 0.20,
        "tokens_evaluated": 100,
        "examples_evaluated": 5,
    }

    report = build_evaluation_report(baseline, quantized)

    assert report["perplexity_comparison"]["change_percent"] == pytest.approx(20.0)
    assert report["accuracy_comparison"]["drop_percentage_points"] == pytest.approx(5.0)
    assert report["accuracy_comparison"]["relative_drop_percent"] == pytest.approx(20.0)


def test_accuracy_improvement_is_not_reported_as_degradation():
    baseline = {
        "perplexity": 10.0,
        "accuracy": 0.20,
    }

    quantized = {
        "perplexity": 9.0,
        "accuracy": 0.25,
    }

    report = build_evaluation_report(baseline, quantized)

    assert report["accuracy_comparison"]["drop_percentage_points"] == pytest.approx(-5.0)