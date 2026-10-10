import pytest
import json

from src.inference_pipeline.config import PipelineConfig
from src.inference_pipeline.deployment import (
    build_vllm_command,
    validate_quantized_model,
)


def test_build_vllm_command(tmp_path):
    config = PipelineConfig(
        artifact_dir=tmp_path,
    )

    command = build_vllm_command(config)

    assert command[0] == "vllm"
    assert command[1] == "serve"
    assert command[2] == str(tmp_path / "quantized_model")

    assert "--host" in command
    assert "--port" in command
    assert "--max-model-len" in command
    assert "--gpu-memory-utilization" in command
    assert "--max-num-seqs" in command


def test_quantization_is_detected_from_model_metadata(tmp_path):
    config = PipelineConfig(artifact_dir=tmp_path)

    command = build_vllm_command(config)

    # The saved model should declare its quantization format.
    assert "--quantization" not in command


def test_invalid_executable_is_rejected():
    with pytest.raises(ValueError):
        build_vllm_command(
            PipelineConfig(),
            executable=" ",
        )


def test_validate_quantized_model(tmp_path):
    model_dir = tmp_path / "quantized_model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        "{}",
        encoding="utf-8",
    )

    validate_quantized_model(model_dir)


def test_missing_model_directory_is_rejected(tmp_path):
    with pytest.raises(FileNotFoundError):
        validate_quantized_model(tmp_path / "missing")


def test_missing_model_config_is_rejected(tmp_path):
    model_dir = tmp_path / "quantized_model"
    model_dir.mkdir()

    with pytest.raises(FileNotFoundError):
        validate_quantized_model(model_dir)

def test_speculative_decoding_is_optional():
    config = PipelineConfig()

    command = build_vllm_command(config)

    assert "--speculative-config" not in command


def test_speculative_decoding_is_added_when_enabled():
    config = PipelineConfig(
        enable_speculative_decoding=True,
        speculative_num_tokens=3,
    )

    command = build_vllm_command(config)

    assert "--speculative-config" in command
    assert "--per-request-spec-decode-metrics" in command

    index = command.index("--speculative-config")
    speculative = json.loads(command[index + 1])

    assert speculative["method"] == "draft_model"
    assert speculative["model"] == config.draft_model_id
    assert speculative["num_speculative_tokens"] == 3