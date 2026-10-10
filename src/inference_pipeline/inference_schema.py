from dataclasses import asdict, dataclass
import math
from statistics import fmean, median
from typing import Optional, Sequence


@dataclass(frozen=True)
class InferenceRecord:
    """Metrics collected from one inference request."""

    request_id: str
    model_variant: str

    prompt_tokens: int
    output_tokens: int

    ttft_ms: float
    latency_ms: float
    itl_ms: Optional[float] = None

    output_text: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate recorded inference metrics."""

        if not self.request_id.strip():
            raise ValueError("Request ID cannot be empty.")

        if not self.model_variant.strip():
            raise ValueError("Model variant cannot be empty.")

        if self.prompt_tokens <= 0:
            raise ValueError("Prompt token count must be positive.")

        if self.output_tokens <= 0:
            raise ValueError("Output token count must be positive.")

        if self.ttft_ms < 0 or self.latency_ms <= 0:
            raise ValueError("Invalid inference timing values.")

        if self.ttft_ms > self.latency_ms:
            raise ValueError("TTFT cannot exceed total latency.")

        timings = [self.ttft_ms, self.latency_ms]

        if self.itl_ms is not None:
            if not math.isfinite(self.itl_ms) or self.itl_ms < 0:
                raise ValueError("ITL must be finite and non-negative.")

        if not all(math.isfinite(value) for value in timings):
            raise ValueError("Timing values must be finite.")

    def to_dict(self) -> dict:
        """Convert the record into a JSON-compatible dictionary."""

        return asdict(self)


def summarize_inference_records(
    records: Sequence[InferenceRecord],
    elapsed_seconds: float,
) -> dict:
    """Summarize requests collected during one benchmark run."""

    if not records:
        raise ValueError("At least one inference record is required.")

    if (
        not math.isfinite(elapsed_seconds)
        or elapsed_seconds <= 0
    ):
        raise ValueError("Benchmark duration must be positive and finite.")

    ttft_values = [record.ttft_ms for record in records]
    latency_values = [record.latency_ms for record in records]

    itl_values = [
        record.itl_ms
        for record in records
        if record.itl_ms is not None
    ]

    total_output_tokens = sum(
        record.output_tokens for record in records
    )

    return {
        "model_variant": records[0].model_variant,
        "request_count": len(records),
        "total_output_tokens": total_output_tokens,
        "mean_ttft_ms": fmean(ttft_values),
        "median_ttft_ms": median(ttft_values),
        "mean_latency_ms": fmean(latency_values),
        "mean_itl_ms": (
            fmean(itl_values) if itl_values else None
        ),
        "tokens_per_second": (
            total_output_tokens / elapsed_seconds
        ),
        "requests_per_second": (
            len(records) / elapsed_seconds
        ),
        "benchmark_duration_seconds": elapsed_seconds,
    }