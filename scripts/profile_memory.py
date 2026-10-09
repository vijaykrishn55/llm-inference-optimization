import json
from pathlib import Path

from src.benchmarking.memory import get_gpu_memory


def save_profile(profile: dict, output_path: str = "results/memory.json"):
    """Save GPU memory profile."""

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(profile, file, indent=2)

    print(f"Memory profile saved to {output_path}")


def main():
    profile = get_gpu_memory()
    save_profile(profile)


if __name__ == "__main__":
    main()