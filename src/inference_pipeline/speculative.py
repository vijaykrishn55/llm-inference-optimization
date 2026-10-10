import math


def build_speculative_config(
    draft_model_id: str,
    num_speculative_tokens: int = 3,
) -> dict:
    """Build vLLM's draft-model speculative decoding configuration."""

    if not draft_model_id.strip():
        raise ValueError("Draft model ID cannot be empty.")

    if num_speculative_tokens <= 0:
        raise ValueError("Speculative token count must be positive.")

    return {
        "method": "draft_model",
        "model": draft_model_id,
        "num_speculative_tokens": num_speculative_tokens,
    }


def calculate_acceptance_rate(
    accepted_draft_tokens: int,
    proposed_draft_tokens: int,
) -> float:
    """Calculate the fraction of proposed draft tokens accepted."""

    if accepted_draft_tokens < 0 or proposed_draft_tokens < 0:
        raise ValueError("Token counts cannot be negative.")

    if accepted_draft_tokens > proposed_draft_tokens:
        raise ValueError(
            "Accepted tokens cannot exceed proposed draft tokens."
        )

    if proposed_draft_tokens == 0:
        return 0.0

    return accepted_draft_tokens / proposed_draft_tokens


def extract_speculative_metrics(response: dict) -> dict:
    """Extract speculative decoding metrics from a vLLM API response."""

    metrics = response.get("metrics", {})
    speculative = metrics.get("speculative_decoding")

    if not isinstance(speculative, dict):
        raise ValueError(
            "Speculative decoding metrics are missing. "
            "Check the server configuration and vLLM version."
        )

    accepted = speculative.get("num_accepted_draft_tokens")
    proposed = speculative.get("num_draft_tokens")

    if not isinstance(accepted, int) or not isinstance(proposed, int):
        raise ValueError("Response contains invalid speculative metrics.")

    return {
        **speculative,
        "calculated_acceptance_rate": calculate_acceptance_rate(
            accepted,
            proposed,
        ),
    }


def compare_generation_speed(
    baseline_tokens_per_second: float,
    speculative_tokens_per_second: float,
) -> dict:
    """Compare baseline and speculative generation speed."""

    values = {
        "baseline": baseline_tokens_per_second,
        "speculative": speculative_tokens_per_second,
    }

    for name, value in values.items():
        if not math.isfinite(value) or value <= 0:
            raise ValueError(
                f"{name.capitalize()} speed must be positive and finite."
            )

    speedup = speculative_tokens_per_second / baseline_tokens_per_second

    return {
        "baseline_tokens_per_second": baseline_tokens_per_second,
        "speculative_tokens_per_second": speculative_tokens_per_second,
        "speedup_factor": speedup,
        "change_percent": (speedup - 1) * 100,
    }