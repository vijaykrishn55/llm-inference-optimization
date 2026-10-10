import math

import pytest

from src.inference_pipeline.evaluation_metrics import (
    perplexity_from_loss,
    calculate_perplexity_change,
    calculate_accuracy_degradation,
)


def test_perplexity_from_loss():
    assert perplexity_from_loss(2.0) == pytest.approx(math.exp(2.0))


def test_perplexity_rejects_negative_loss():
    with pytest.raises(ValueError):
        perplexity_from_loss(-1.0)


def test_perplexity_rejects_non_finite_loss():
    with pytest.raises(ValueError):
        perplexity_from_loss(float("nan"))


def test_perplexity_change():
    result = calculate_perplexity_change(2.0, 2.2)

    assert result["change"] == pytest.approx(0.2)
    assert result["change_percent"] == pytest.approx(10.0)


def test_perplexity_change_rejects_zero_baseline():
    with pytest.raises(ValueError):
        calculate_perplexity_change(0.0, 2.0)


def test_accuracy_degradation():
    result = calculate_accuracy_degradation(0.90, 0.87)

    assert result["drop_percentage_points"] == pytest.approx(3.0)
    assert result["relative_drop_percent"] == pytest.approx(3.333333)


def test_accuracy_improvement_returns_negative_drop():
    result = calculate_accuracy_degradation(0.80, 0.84)

    assert result["drop_percentage_points"] == pytest.approx(-4.0)


def test_accuracy_rejects_values_outside_range():
    with pytest.raises(ValueError):
        calculate_accuracy_degradation(1.1, 0.9)