from pathlib import Path


def get_directory_size(path: str | Path) -> int:
    """Return total size of all files inside a directory."""

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")

    total_size = 0

    for file in path.rglob("*"):
        if file.is_file():
            total_size += file.stat().st_size

    return total_size


def format_size(size_bytes: int) -> str:
    """Convert bytes to a human-readable size."""

    size = float(size_bytes)

    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024:
            return f"{size:.2f} {unit}"

        size /= 1024

    return f"{size:.2f} PB"


def calculate_reduction(
    original_bytes: int,
    quantized_bytes: int,
) -> dict:

    reduction = (
        (original_bytes - quantized_bytes)
        / original_bytes
    ) * 100

    compression_ratio = (
        original_bytes / quantized_bytes
    )

    return {
        "original_bytes": original_bytes,
        "quantized_bytes": quantized_bytes,
        "reduction_percent": reduction,
        "compression_ratio": compression_ratio,
    }