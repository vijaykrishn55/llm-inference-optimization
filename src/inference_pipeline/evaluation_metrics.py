import math


def perplexity_from_loss(mean_loss: float) -> float:
    """Convert mean cross-entropy loss into perplexity."""

    if not math.isfinite(mean_loss) or mean_loss < 0:
        raise ValueError("Mean loss must be finite and non-negative.")

    try:
        return math.exp(mean_loss)
    except OverflowError as exc:
        raise ValueError("Mean loss is too large.") from exc


def calculate_perplexity_change(
    baseline_perplexity: float,
    quantized_perplexity: float,
) -> dict:
    """Compare quantized perplexity against the FP16 baseline."""

    if (
        not math.isfinite(baseline_perplexity)
        or baseline_perplexity <= 0
    ):
        raise ValueError("Baseline perplexity must be positive and finite.")

    if (
        not math.isfinite(quantized_perplexity)
        or quantized_perplexity <= 0
    ):
        raise ValueError("Quantized perplexity must be positive and finite.")

    change = quantized_perplexity - baseline_perplexity

    return {
        "baseline_perplexity": baseline_perplexity,
        "quantized_perplexity": quantized_perplexity,
        "change": change,
        "change_percent": change / baseline_perplexity * 100,
    }


def calculate_accuracy_degradation(
    baseline_accuracy: float,
    quantized_accuracy: float,
) -> dict:
    """Calculate the accuracy change after quantization.

    Accuracy values must be fractions between 0 and 1.
    A negative drop means accuracy improved after quantization.
    """

    for name, value in (
        ("Baseline accuracy", baseline_accuracy),
        ("Quantized accuracy", quantized_accuracy),
    ):
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{name} must be between 0 and 1.")

    if baseline_accuracy == 0:
        raise ValueError(
            "Relative degradation is undefined when baseline accuracy is zero."
        )

    drop = baseline_accuracy - quantized_accuracy

    return {
        "baseline_accuracy": baseline_accuracy,
        "quantized_accuracy": quantized_accuracy,
        "drop_percentage_points": drop * 100,
        "relative_drop_percent": drop / baseline_accuracy * 100,
    }