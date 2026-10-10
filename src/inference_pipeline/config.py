from dataclasses import dataclass
from pathlib import Path
from typing import Literal


QuantizationMethod = Literal["awq", "gptq"]


@dataclass(frozen=True)
class PipelineConfig:
    """Central configuration for the LLM inference pipeline."""

    target_model_id: str = "Qwen/Qwen2.5-1.5B-Instruct"
    draft_model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"

    quantization_method: QuantizationMethod = "awq"
    quantization_scheme: str = "W4A16_ASYM"

    calibration_dataset: str = "wikitext"
    calibration_samples: int = 64

    max_sequence_length: int = 512
    max_model_length: int = 2048

    server_host: str = "127.0.0.1"
    server_port: int = 8000

    benchmark_requests: int = 10
    benchmark_concurrency: int = 1
    prompt_tokens: int = 128
    generation_tokens: int = 128

    gpu_memory_utilization: float = 0.85
    max_num_seqs: int = 16

    enable_speculative_decoding: bool = False
    speculative_num_tokens: int = 3

    artifact_dir: Path = Path("artifacts")
    report_dir: Path = Path("reports")

    def __post_init__(self) -> None:
        """Validate configuration values."""

        if not self.target_model_id.strip():
            raise ValueError("Target model ID cannot be empty.")

        if not self.draft_model_id.strip():
            raise ValueError("Draft model ID cannot be empty.")

        if self.quantization_method not in ("awq", "gptq"):
            raise ValueError(
                "Quantization method must be 'awq' or 'gptq'."
            )

        if self.quantization_scheme not in ("W4A16_ASYM", "W8A16"):
            raise ValueError(
                "Quantization scheme must be W4A16_ASYM or W8A16."
            )

        if self.calibration_samples <= 0:
            raise ValueError("Calibration samples must be positive.")

        if self.max_sequence_length <= 0:
            raise ValueError("Maximum sequence length must be positive.")

        if self.max_model_length < self.max_sequence_length:
            raise ValueError(
                "Maximum model length must be at least "
                "the maximum sequence length."
            )

        if not 1 <= self.server_port <= 65535:
            raise ValueError("Server port must be between 1 and 65535.")

        if self.benchmark_requests <= 0:
            raise ValueError("Benchmark requests must be positive.")

        if self.benchmark_concurrency <= 0:
            raise ValueError("Benchmark concurrency must be positive.")

        if self.prompt_tokens <= 0 or self.generation_tokens <= 0:
            raise ValueError(
                "Prompt and generation token counts must be positive."
            )

        if (
            self.prompt_tokens + self.generation_tokens
            > self.max_model_length
        ):
            raise ValueError(
                "Prompt and generation tokens exceed maximum model length."
            )

        if not 0 < self.gpu_memory_utilization <= 1:
            raise ValueError(
                "GPU memory utilization must be greater than 0 and at most 1."
            )

        if self.max_num_seqs <= 0:
            raise ValueError("Maximum number of sequences must be positive.")

        if self.speculative_num_tokens <= 0:
            raise ValueError(
                "Speculative token count must be positive."
            )
        
    @property
    def quantized_model_dir(self) -> Path:
        """Return the destination for the quantized model."""
        return self.artifact_dir / "quantized_model"

    @property
    def benchmark_report_path(self) -> Path:
        """Return the destination for benchmark results."""
        return self.report_dir / "benchmark.json"

    @property
    def fp16_model_dir(self) -> Path:
        """Return the destination for the original FP16 model."""
        return self.artifact_dir / "fp16_model"