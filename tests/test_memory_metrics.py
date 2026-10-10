import pytest

from src.inference_pipeline.memory_metrics import (
    estimate_kv_cache_bytes,
    bytes_to_gib,
    summarize_memory_usage,
    calculate_memory_reduction,
)


def test_estimate_kv_cache_bytes():
    result = estimate_kv_cache_bytes(
        num_layers=2,
        batch_size=1,
        sequence_length=1024,
        num_kv_heads=4,
        head_dim=64,
        bytes_per_element=2,
    )

    assert result == 2_097_152


def test_kv_cache_rejects_invalid_dimensions():
    with pytest.raises(ValueError):
        estimate_kv_cache_bytes(
            num_layers=0,
            batch_size=1,
            sequence_length=1024,
            num_kv_heads=4,
            head_dim=64,
        )


def test_bytes_to_gib():
    assert bytes_to_gib(1024 ** 3) == 1.0


def test_bytes_to_gib_rejects_negative_size():
    with pytest.raises(ValueError):
        bytes_to_gib(-1)


def test_summarize_memory_usage():
    result = summarize_memory_usage(
        weight_bytes=1_000,
        kv_cache_bytes=2_000,
        activation_bytes=500,
    )

    assert result["total_bytes"] == 3_500
    assert result["weight_bytes"] == 1_000
    assert result["kv_cache_bytes"] == 2_000
    assert result["activation_bytes"] == 500


def test_memory_summary_rejects_negative_components():
    with pytest.raises(ValueError):
        summarize_memory_usage(
            weight_bytes=1_000,
            kv_cache_bytes=-1,
            activation_bytes=500,
        )


def test_calculate_memory_reduction():
    result = calculate_memory_reduction(
        baseline_bytes=1_000,
        optimized_bytes=600,
    )

    assert result["reduction_bytes"] == 400
    assert result["reduction_percent"] == pytest.approx(40.0)


def test_memory_reduction_rejects_zero_baseline():
    with pytest.raises(ValueError):
        calculate_memory_reduction(
            baseline_bytes=0,
            optimized_bytes=500,
        )