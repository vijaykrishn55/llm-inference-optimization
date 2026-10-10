import csv
import io
import re
import subprocess
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from src.inference_pipeline.memory_metrics import estimate_kv_cache_bytes
from src.inference_pipeline.size_metrics import measure_directory_size


KV_CACHE_PATTERN = re.compile(
    r"^(vllm:(?:kv_cache_usage_perc|gpu_cache_usage_perc))"
    r"(?:\{[^}]*\})?\s+([-+]?(?:\d+(?:\.\d*)?|\.\d+)"
    r"(?:[eE][-+]?\d+)?)\s*$"
)


def parse_nvidia_smi_output(output: str) -> list[dict]:
    """Parse GPU memory information returned by nvidia-smi."""
    results = []

    for row in csv.reader(io.StringIO(output)):
        if not row or not any(value.strip() for value in row):
            continue

        if len(row) != 5:
            raise ValueError(f"Unexpected nvidia-smi row: {row}")

        try:
            results.append({
                "gpu_name": row[0].strip(),
                "memory_total_mib": float(row[1].strip()),
                "memory_used_mib": float(row[2].strip()),
                "memory_free_mib": float(row[3].strip()),
                "gpu_utilization_percent": float(row[4].strip()),
            })
        except ValueError as exc:
            raise ValueError(f"Invalid GPU statistics row: {row}") from exc

    if not results:
        raise ValueError("No GPU statistics found in nvidia-smi output.")

    return results


def collect_gpu_stats(executable: str = "nvidia-smi") -> list[dict]:
    """Collect current GPU memory and utilization from NVIDIA's utility."""
    command = [
        executable,
        "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError("Could not execute nvidia-smi.") from exc

    if result.returncode != 0:
        raise RuntimeError(
            f"nvidia-smi failed: {result.stderr.strip()}"
        )

    return parse_nvidia_smi_output(result.stdout)


def parse_vllm_metrics(metrics_text: str) -> dict:
    """Extract KV-cache utilization from vLLM Prometheus metrics."""
    values = []
    metric_name = None

    for line in metrics_text.splitlines():
        match = KV_CACHE_PATTERN.match(line.strip())

        if match:
            metric_name = match.group(1)
            values.append(float(match.group(2)))

    return {
        "kv_cache_usage_fraction": max(values) if values else None,
        "metric_name": metric_name,
        "samples": values,
    }


def fetch_vllm_metrics(
    base_url: str,
    timeout: float = 5.0,
) -> dict:
    """Fetch KV-cache metrics from a running vLLM server."""
    if not base_url.strip():
        raise ValueError("Base URL cannot be empty.")
    if timeout <= 0:
        raise ValueError("Timeout must be positive.")

    request = Request(
        f"{base_url.rstrip('/')}/metrics",
        headers={"Accept": "text/plain"},
    )

    try:
        with urlopen(request, timeout=timeout) as response:
            metrics_text = response.read().decode("utf-8")
    except HTTPError as exc:
        raise RuntimeError(
            f"vLLM metrics endpoint returned HTTP {exc.code}."
        ) from exc
    except (URLError, TimeoutError) as exc:
        raise RuntimeError(
            f"Could not reach vLLM metrics endpoint: {exc}"
        ) from exc

    return parse_vllm_metrics(metrics_text)


def estimate_activation_memory_bytes(
    batch_size: int,
    sequence_length: int,
    hidden_size: int,
    intermediate_size: int,
    bytes_per_element: int = 2,
) -> int:
    """Estimate major temporary activation tensors.

    This is a rough estimate, not a measurement of vLLM's actual
    peak activation memory.
    """
    values = (
        batch_size,
        sequence_length,
        hidden_size,
        intermediate_size,
        bytes_per_element,
    )

    if any(value <= 0 for value in values):
        raise ValueError("All activation dimensions must be positive.")

    return (
        batch_size
        * sequence_length
        * (3 * hidden_size + intermediate_size)
        * bytes_per_element
    )


def build_memory_profile(
    model_dir: str | Path,
    gpu_stats: list[dict],
    num_layers: int,
    hidden_size: int,
    intermediate_size: int,
    num_kv_heads: int,
    head_dim: int,
    batch_size: int,
    sequence_length: int,
    bytes_per_element: int = 2,
    vllm_metrics: dict | None = None,
) -> dict:
    """Combine artifact size, memory estimates, and live GPU statistics."""
    model_path = Path(model_dir)

    if not model_path.is_dir():
        raise NotADirectoryError(f"Model directory not found: {model_path}")
    if not gpu_stats:
        raise ValueError("At least one GPU statistics record is required.")

    artifact_bytes = measure_directory_size(model_path)

    kv_cache_bytes = estimate_kv_cache_bytes(
        num_layers=num_layers,
        batch_size=batch_size,
        sequence_length=sequence_length,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        bytes_per_element=bytes_per_element,
    )

    activation_bytes = estimate_activation_memory_bytes(
        batch_size=batch_size,
        sequence_length=sequence_length,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        bytes_per_element=bytes_per_element,
    )

    return {
        "gpu_stats": gpu_stats,
        "vllm_metrics": vllm_metrics,
        "configuration": {
            "batch_size": batch_size,
            "sequence_length": sequence_length,
            "bytes_per_element": bytes_per_element,
        },
        "memory_components": {
            "model_artifact_bytes": artifact_bytes,
            "kv_cache_estimate_bytes": kv_cache_bytes,
            "activation_estimate_bytes": activation_bytes,
            "estimated_component_sum_bytes": (
                artifact_bytes + kv_cache_bytes + activation_bytes
            ),
        },
        "notes": [
            "GPU memory is measured with nvidia-smi at profiling time.",
            "Model artifact size is an on-disk measurement, not exact GPU weight memory.",
            "KV-cache and activation values are estimates, not measured allocations.",
            "GPU memory also includes runtime overhead and other allocations.",
        ],
    }