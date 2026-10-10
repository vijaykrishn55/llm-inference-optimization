from collections.abc import Sequence
import math


def calculate_ttft_ms(
    request_start_time: float,
    first_token_time: float,
) -> float:
    """Calculate time to first token in milliseconds."""

    duration = first_token_time - request_start_time

    if not math.isfinite(duration) or duration < 0:
        raise ValueError("First token time cannot precede request start.")

    return duration * 1000


def calculate_inter_token_latency_ms(
    token_timestamps: Sequence[float],
) -> float:
    """Calculate average time between generated tokens in milliseconds."""

    if len(token_timestamps) < 2:
        raise ValueError("At least two token timestamps are required.")

    if not all(math.isfinite(t) for t in token_timestamps):
        raise ValueError("Token timestamps must be finite numbers.")

    intervals = [
        current - previous
        for previous, current in zip(
            token_timestamps,
            token_timestamps[1:],
        )
    ]

    if any(interval < 0 for interval in intervals):
        raise ValueError("Token timestamps must be chronological.")

    return sum(intervals) / len(intervals) * 1000


def calculate_tokens_per_second(
    output_tokens: int,
    generation_time_seconds: float,
) -> float:
    """Calculate single-request generation speed."""

    if output_tokens <= 0:
        raise ValueError("Output token count must be positive.")

    if (
        not math.isfinite(generation_time_seconds)
        or generation_time_seconds <= 0
    ):
        raise ValueError("Generation time must be positive and finite.")

    return output_tokens / generation_time_seconds


def calculate_request_throughput(
    completed_requests: int,
    duration_seconds: float,
) -> float:
    """Calculate completed requests per second."""

    if completed_requests <= 0:
        raise ValueError("Completed requests must be positive.")

    if not math.isfinite(duration_seconds) or duration_seconds <= 0:
        raise ValueError("Duration must be positive and finite.")

    return completed_requests / duration_seconds


def calculate_token_throughput(
    total_output_tokens: int,
    duration_seconds: float,
) -> float:
    """Calculate aggregate output tokens per second."""

    if total_output_tokens <= 0:
        raise ValueError("Total output tokens must be positive.")

    if not math.isfinite(duration_seconds) or duration_seconds <= 0:
        raise ValueError("Duration must be positive and finite.")

    return total_output_tokens / duration_seconds