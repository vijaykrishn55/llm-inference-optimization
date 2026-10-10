import pytest

from src.inference_pipeline.benchmark_metrics import (
    calculate_ttft_ms,
    calculate_inter_token_latency_ms,
    calculate_tokens_per_second,
    calculate_request_throughput,
    calculate_token_throughput,
)


def test_calculate_ttft():
    assert calculate_ttft_ms(10.0, 10.25) == pytest.approx(250.0)


def test_ttft_rejects_negative_duration():
    with pytest.raises(ValueError):
        calculate_ttft_ms(10.0, 9.0)


def test_calculate_inter_token_latency():
    timestamps = [1.0, 1.1, 1.3]

    assert calculate_inter_token_latency_ms(
        timestamps
    ) == pytest.approx(150.0)


def test_itl_requires_two_tokens():
    with pytest.raises(ValueError):
        calculate_inter_token_latency_ms([1.0])


def test_itl_rejects_unordered_timestamps():
    with pytest.raises(ValueError):
        calculate_inter_token_latency_ms([1.0, 1.3, 1.2])


def test_calculate_tokens_per_second():
    assert calculate_tokens_per_second(100, 2.0) == 50.0


def test_tokens_per_second_rejects_zero_duration():
    with pytest.raises(ValueError):
        calculate_tokens_per_second(100, 0.0)


def test_calculate_request_throughput():
    assert calculate_request_throughput(10, 5.0) == 2.0


def test_calculate_token_throughput():
    assert calculate_token_throughput(1000, 5.0) == 200.0


def test_throughput_rejects_zero_duration():
    with pytest.raises(ValueError):
        calculate_token_throughput(1000, 0.0)