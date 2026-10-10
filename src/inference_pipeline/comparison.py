import math


def compare_metric_values(
    baseline_value: float | int | None,
    optimized_value: float | int | None,
    higher_is_better: bool,
) -> dict:
    """Compare one metric and calculate its directional improvement."""

    result = {
        "baseline": baseline_value,
        "optimized": optimized_value,
        "change_percent": None,
        "improvement_percent": None,
    }

    if baseline_value is None or optimized_value is None:
        return result

    values = (baseline_value, optimized_value)
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        for value in values
    ):
        raise ValueError("Metric values must be finite numbers.")

    if baseline_value == 0:
        if optimized_value == 0:
            result["change_percent"] = 0.0
            result["improvement_percent"] = 0.0
        return result

    change = (
        (optimized_value - baseline_value) / abs(baseline_value)
    ) * 100

    result["change_percent"] = round(change, 3)
    result["improvement_percent"] = round(
        change if higher_is_better else -change, 3
    )

    return result


def calculate_failure_rate(report: dict) -> float | None:
    """Calculate failed requests as a percentage of requested requests."""
    results = report.get("results", {})
    requested = results.get("requests_requested", 0)
    failed = results.get("requests_failed", 0)

    if requested <= 0:
        return None

    if not 0 <= failed <= requested:
        raise ValueError("Failed requests must be between zero and total requests.")

    return failed / requested * 100


def compare_benchmark_reports(
    baseline_report: dict,
    optimized_report: dict,
) -> dict:
    """Compare performance metrics from two benchmark reports."""

    baseline = baseline_report.get("results", {})
    optimized = optimized_report.get("results", {})

    metric_directions = {
        "mean_ttft_ms": False,
        "median_ttft_ms": False,
        "mean_itl_ms": False,
        "mean_latency_ms": False,
        "request_throughput_per_second": True,
        "token_throughput_per_second": True,
    }

    comparisons = {}

    for metric, higher_is_better in metric_directions.items():
        comparisons[metric] = compare_metric_values(
            baseline.get(metric),
            optimized.get(metric),
            higher_is_better,
        )

    comparisons["failure_rate_percent"] = compare_metric_values(
        calculate_failure_rate(baseline_report),
        calculate_failure_rate(optimized_report),
        higher_is_better=False,
    )

    return {
        "baseline_model": baseline_report.get("settings", {}).get("model"),
        "optimized_model": optimized_report.get("settings", {}).get("model"),
        "metric_comparisons": comparisons,
        "notes": [
            "Positive improvement means the metric moved in the preferred direction.",
            "Lower TTFT, ITL, latency, and failure rate are better.",
            "Higher request and token throughput are better.",
            "A percentage change is undefined when the baseline is zero and the optimized value is nonzero.",
        ],
    }