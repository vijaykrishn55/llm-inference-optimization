import shutil
import subprocess
import json

from src.inference_pipeline.speculative import build_speculative_config
from pathlib import Path

from src.inference_pipeline.config import PipelineConfig


def validate_quantized_model(model_path: Path) -> None:
    """Verify that a saved model directory is ready to be served."""

    if not model_path.is_dir():
        raise FileNotFoundError(
            f"Quantized model directory not found: {model_path}"
        )

    if not (model_path / "config.json").is_file():
        raise FileNotFoundError(
            f"Model configuration not found in: {model_path}"
        )


def build_vllm_command(
    config: PipelineConfig,
    executable: str = "vllm",
) -> list[str]:
    """Build the vLLM OpenAI-compatible server command."""

    if not executable.strip():
        raise ValueError("vLLM executable cannot be empty.")

    command = [
        executable,
        "serve",
        str(config.quantized_model_dir),
        "--host",
        config.server_host,
        "--port",
        str(config.server_port),
        "--max-model-len",
        str(config.max_model_length),
        "--gpu-memory-utilization",
        str(config.gpu_memory_utilization),
        "--max-num-seqs",
        str(config.max_num_seqs),
    ]

    if config.enable_speculative_decoding:
        speculative_config = build_speculative_config(
            draft_model_id=config.draft_model_id,
            num_speculative_tokens=config.speculative_num_tokens,
        )

        command.extend([
            "--speculative-config",
            json.dumps(speculative_config),
            "--per-request-spec-decode-metrics",
            "summary",
        ])

    return command


def launch_vllm_server(config: PipelineConfig) -> None:
    """Launch vLLM and keep the server running in the foreground."""

    model_path = config.quantized_model_dir
    validate_quantized_model(model_path)

    executable = shutil.which("vllm")

    if executable is None:
        raise RuntimeError(
            "vLLM CLI was not found. Install a compatible vLLM build "
            "in the Kaggle GPU environment before launching the server."
        )

    command = build_vllm_command(
        config,
        executable=executable,
    )

    subprocess.run(
        command,
        check=True,
    )