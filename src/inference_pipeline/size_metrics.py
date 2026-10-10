from pathlib import Path
from typing import Union


PathLike = Union[str, Path]


def measure_directory_size(path: PathLike) -> int:
    """Return the total size of all files in a directory, in bytes."""

    directory = Path(path)

    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    return sum(
        file.stat().st_size
        for file in directory.rglob("*")
        if file.is_file()
    )


def compare_model_sizes(
    original_path: PathLike,
    quantized_path: PathLike,
) -> dict:
    """Compare original and quantized model directory sizes."""

    original_bytes = measure_directory_size(original_path)
    quantized_bytes = measure_directory_size(quantized_path)

    if original_bytes == 0:
        raise ValueError("Original model directory contains no data.")

    reduction_bytes = original_bytes - quantized_bytes
    reduction_percent = reduction_bytes / original_bytes * 100

    return {
        "original_bytes": original_bytes,
        "quantized_bytes": quantized_bytes,
        "reduction_bytes": reduction_bytes,
        "reduction_percent": reduction_percent,
    }


def bytes_to_mb(size_bytes: int) -> float:
    """Convert bytes to megabytes using decimal units."""

    return size_bytes / 1_000_000