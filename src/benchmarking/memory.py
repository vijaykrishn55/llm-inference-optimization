import torch


def get_gpu_memory() -> dict:
    """Return current CUDA memory statistics."""

    if not torch.cuda.is_available():
        return {
            "available": False,
        }

    device = torch.cuda.current_device()

    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    peak_allocated = torch.cuda.max_memory_allocated(device)

    total = torch.cuda.get_device_properties(device).total_memory

    return {
        "available": True,
        "device": torch.cuda.get_device_name(device),
        "allocated_gb": allocated / (1024 ** 3),
        "reserved_gb": reserved / (1024 ** 3),
        "peak_allocated_gb": peak_allocated / (1024 ** 3),
        "total_gb": total / (1024 ** 3),
    }