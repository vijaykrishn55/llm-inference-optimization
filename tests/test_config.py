import pytest

from src.inference_pipeline.config import PipelineConfig


def test_default_configuration():
    config = PipelineConfig()

    assert config.quantization_method == "awq"
    assert config.quantization_scheme == "W4A16_ASYM"
    assert config.calibration_samples == 64
    assert config.server_port == 8000


def test_model_ids_are_configured():
    config = PipelineConfig()

    assert "1.5B" in config.target_model_id
    assert "0.5B" in config.draft_model_id


def test_invalid_quantization_method():
    with pytest.raises(ValueError):
        PipelineConfig(quantization_method="invalid")


def test_calibration_samples_must_be_positive():
    with pytest.raises(ValueError):
        PipelineConfig(calibration_samples=0)


def test_model_length_must_cover_sequence_length():
    with pytest.raises(ValueError):
        PipelineConfig(
            max_sequence_length=1024,
            max_model_length=512,
        )


def test_server_port_must_be_valid():
    with pytest.raises(ValueError):
        PipelineConfig(server_port=70000)


def test_benchmark_token_lengths_must_fit():
    with pytest.raises(ValueError):
        PipelineConfig(
            prompt_tokens=1500,
            generation_tokens=1000,
            max_model_length=2048,
        )


def test_output_paths():
    config = PipelineConfig()

    assert config.quantized_model_dir == (
        config.artifact_dir / "quantized_model"
    )
    assert config.benchmark_report_path == (
        config.report_dir / "benchmark.json"
    )

def test_gpu_memory_utilization_must_be_valid():
    with pytest.raises(ValueError):
        PipelineConfig(gpu_memory_utilization=0.0)

    with pytest.raises(ValueError):
        PipelineConfig(gpu_memory_utilization=1.1)


def test_max_num_seqs_must_be_positive():
    with pytest.raises(ValueError):
        PipelineConfig(max_num_seqs=0)

def test_speculative_token_count_must_be_positive():
    with pytest.raises(ValueError):
        PipelineConfig(speculative_num_tokens=0)


def test_speculative_decoding_is_disabled_by_default():
    config = PipelineConfig()

    assert config.enable_speculative_decoding is False
    assert config.speculative_num_tokens == 3