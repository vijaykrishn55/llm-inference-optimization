import pytest

from src.inference_pipeline.speculative import (
    build_speculative_config,
    calculate_acceptance_rate,
    extract_speculative_metrics,
    compare_generation_speed,
)


def test_build_speculative_config():
    result = build_speculative_config(
        "Qwen/Qwen2.5-0.5B-Instruct",
        num_speculative_tokens=3,
    )

    assert result == {
        "method": "draft_model",
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "num_speculative_tokens": 3,
    }


def test_empty_draft_model_is_rejected():
    with pytest.raises(ValueError):
        build_speculative_config(" ")


def test_speculative_token_count_must_be_positive():
    with pytest.raises(ValueError):
        build_speculative_config("draft-model", 0)


def test_calculate_acceptance_rate():
    assert calculate_acceptance_rate(75, 100) == pytest.approx(0.75)


def test_zero_proposals_returns_zero_acceptance():
    assert calculate_acceptance_rate(0, 0) == 0.0


def test_accepted_tokens_cannot_exceed_proposals():
    with pytest.raises(ValueError):
        calculate_acceptance_rate(11, 10)


def test_extract_speculative_metrics():
    response = {
        "metrics": {
            "speculative_decoding": {
                "num_accepted_draft_tokens": 60,
                "num_draft_tokens": 100,
                "draft_acceptance_rate": 0.60,
                "mean_acceptance_length": 1.8,
            }
        }
    }

    result = extract_speculative_metrics(response)

    assert result["calculated_acceptance_rate"] == pytest.approx(0.60)


def test_missing_speculative_metrics_are_rejected():
    with pytest.raises(ValueError, match="metrics are missing"):
        extract_speculative_metrics({"choices": []})


def test_compare_generation_speed():
    result = compare_generation_speed(40.0, 50.0)

    assert result["speedup_factor"] == pytest.approx(1.25)
    assert result["change_percent"] == pytest.approx(25.0)


def test_invalid_generation_speed_is_rejected():
    with pytest.raises(ValueError):
        compare_generation_speed(0.0, 50.0)