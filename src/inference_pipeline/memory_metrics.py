def estimate_kv_cache_bytes(
    num_layers: int,
    batch_size: int,
    sequence_length: int,
    num_kv_heads: int,
    head_dim: int,
    bytes_per_element: int = 2,
) -> int:
    """Estimate KV cache memory for a decoder-only transformer.

    Assumes the key and value caches use the same precision.
    """

    values = {
        "num_layers": num_layers,
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "bytes_per_element": bytes_per_element,
    }

    for name, value in values.items():
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer.")

    return (
        2
        * num_layers
        * batch_size
        * sequence_length
        * num_kv_heads
        * head_dim
        * bytes_per_element
    )


def bytes_to_gib(size_bytes: int) -> float:
    """Convert bytes to GiB (1024³ bytes)."""

    if not isinstance(size_bytes, int) or size_bytes < 0:
        raise ValueError("Memory size must be a non-negative integer.")

    return size_bytes / (1024 ** 3)


def summarize_memory_usage(
    weight_bytes: int,
    kv_cache_bytes: int,
    activation_bytes: int,
) -> dict:
    """Summarize memory components from measurements or estimates."""

    components = {
        "weight_bytes": weight_bytes,
        "kv_cache_bytes": kv_cache_bytes,
        "activation_bytes": activation_bytes,
    }

    for name, value in components.items():
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer.")

    total_bytes = sum(components.values())

    return {
        **components,
        "total_bytes": total_bytes,
        "weight_gib": bytes_to_gib(weight_bytes),
        "kv_cache_gib": bytes_to_gib(kv_cache_bytes),
        "activation_gib": bytes_to_gib(activation_bytes),
        "total_gib": bytes_to_gib(total_bytes),
    }


def calculate_memory_reduction(
    baseline_bytes: int,
    optimized_bytes: int,
) -> dict:
    """Compare baseline and optimized memory consumption."""

    if baseline_bytes <= 0:
        raise ValueError("Baseline memory must be positive.")

    if optimized_bytes < 0:
        raise ValueError("Optimized memory cannot be negative.")

    reduction_bytes = baseline_bytes - optimized_bytes

    return {
        "baseline_bytes": baseline_bytes,
        "optimized_bytes": optimized_bytes,
        "reduction_bytes": reduction_bytes,
        "reduction_percent": reduction_bytes / baseline_bytes * 100,
    }