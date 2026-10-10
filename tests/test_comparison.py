import pytest

from src.inference_pipeline.comparison import (
    calculate_failure_rate,
    compare_benchmark_reports,
    compare_metric_values,
)


def test_lower_latency_is_an_improvement():
    result = compare_metric_values(100, 80, higher_is_better=False)

    assert result["change_percent"] == -20.0
    assert result["improvement_percent"] == 20.0


def test_higher_throughput_is_an_improvement():
    result = compare_metric_values(50, 75, higher_is_better=True)

    assert result["change_percent"] == 50.0
    assert result["improvement_percent"] == 50.0


def test_missing_metric_returns_unavailable_percentages():
    result = compare_metric_values(20, None, higher_is_better=False)

    assert result["baseline"] == 20
    assert result["optimized"] is None
    assert result["improvement_percent"] is None


def test_zero_baseline_is_handled():
    result = compare_metric_values(0, 5, higher_is_better=True)
    assert result["change_percent"] is None
    assert result["improvement_percent"] is None


def test_invalid_metric_is_rejected():
    with pytest.raises(ValueError):
        compare_metric_values(float("inf"), 10, higher_is_better=True)


def test_calculate_failure_rate():
    report = {
        "results": {
            "requests_requested": 10,
            "requests_failed": 2,
        }
    }

    assert calculate_failure_rate(report) == 20.0


def test_failure_rate_handles_zero_requests():
    assert calculate_failure_rate({"results": {}}) is None


def test_compare_benchmark_reports():
    baseline = {
        "settings": {"model": "fp16-model"},
        "results": {
            "requests_requested": 10,
            "requests_failed": 1,
            "mean_ttft_ms": 100,
            "median_ttft_ms": 90,
            "mean_itl_ms": 20,
            "mean_latency_ms": 500,
            "request_throughput_per_second": 2,
            "token_throughput_per_second": 100,
        },
    }

    optimized = {
        "settings": {"model": "quantized-model"},
        "results": {
            "requests_requested": 10,
            "requests_failed": 0,
            "mean_ttft_ms": 80,
            "median_ttft_ms": 70,
            "mean_itl_ms": 15,
            "mean_latency_ms": 400,
            "request_throughput_per_second": 3,
            "token_throughput_per_second": 150,
        },
    }

    result = compare_benchmark_reports(baseline, optimized)
    comparisons = result["metric_comparisons"]

    assert result["baseline_model"] == "fp16-model"
    assert result["optimized_model"] == "quantized-model"
    assert comparisons["mean_ttft_ms"]["improvement_percent"] == 20.0
    assert comparisons["token_throughput_per_second"]["improvement_percent"] == 50.0
    assert comparisons["failure_rate_percent"]["improvement_percent"] == 100.0