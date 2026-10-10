import pytest

from src.inference_pipeline.profiling import (
    build_memory_profile,
    estimate_activation_memory_bytes,
    parse_nvidia_smi_output,
    parse_vllm_metrics,
)


def test_parse_nvidia_smi_output():
    output = "Tesla T4, 15360, 4096, 11264, 35\n"
    result = parse_nvidia_smi_output(output)

    assert len(result) == 1
    assert result[0]["gpu_name"] == "Tesla T4"
    assert result[0]["memory_total_mib"] == 15360
    assert result[0]["memory_used_mib"] == 4096
    assert result[0]["memory_free_mib"] == 11264
    assert result[0]["gpu_utilization_percent"] == 35


def test_parse_nvidia_smi_rejects_invalid_output():
    with pytest.raises(ValueError):
        parse_nvidia_smi_output("invalid row")


def test_parse_vllm_metrics():
    metrics = """
# HELP vllm:kv_cache_usage_perc KV cache usage
vllm:kv_cache_usage_perc{engine="0"} 0.25
vllm:kv_cache_usage_perc{engine="1"} 0.60
"""
    result = parse_vllm_metrics(metrics)

    assert result["kv_cache_usage_fraction"] == 0.60
    assert result["samples"] == [0.25, 0.60]


def test_parse_vllm_metrics_returns_none_when_missing():
    result = parse_vllm_metrics("# No KV cache metrics available")
    assert result["kv_cache_usage_fraction"] is None
    assert result["samples"] == []


def test_estimate_activation_memory():
    result = estimate_activation_memory_bytes(
        batch_size=1,
        sequence_length=10,
        hidden_size=8,
        intermediate_size=16,
        bytes_per_element=2,
    )
    assert result == 10 * (3 * 8 + 16) * 2


def test_estimate_activation_memory_rejects_invalid_dimensions():
    with pytest.raises(ValueError):
        estimate_activation_memory_bytes(1, 0, 8, 16)


def test_build_memory_profile(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "weights.bin").write_bytes(b"12345678")

    gpu_stats = [{
        "gpu_name": "Test GPU",
        "memory_total_mib": 1000,
        "memory_used_mib": 200,
        "memory_free_mib": 800,
        "gpu_utilization_percent": 25,
    }]

    profile = build_memory_profile(
        model_dir=model_dir,
        gpu_stats=gpu_stats,
        num_layers=2,
        hidden_size=8,
        intermediate_size=16,
        num_kv_heads=2,
        head_dim=4,
        batch_size=1,
        sequence_length=10,
        bytes_per_element=2,
        vllm_metrics={"kv_cache_usage_fraction": 0.25},
    )

    components = profile["memory_components"]
    assert components["model_artifact_bytes"] == 8
    assert components["kv_cache_estimate_bytes"] > 0
    assert components["activation_estimate_bytes"] > 0
    assert components["estimated_component_sum_bytes"] == (
        components["model_artifact_bytes"]
        + components["kv_cache_estimate_bytes"]
        + components["activation_estimate_bytes"]
    )
    assert profile["gpu_stats"] == gpu_stats


def test_build_memory_profile_requires_gpu_stats(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    with pytest.raises(ValueError, match="GPU statistics"):
        build_memory_profile(
            model_dir=model_dir,
            gpu_stats=[],
            num_layers=2,
            hidden_size=8,
            intermediate_size=16,
            num_kv_heads=2,
            head_dim=4,
            batch_size=1,
            sequence_length=10,
        )