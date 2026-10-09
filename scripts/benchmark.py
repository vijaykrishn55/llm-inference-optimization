import subprocess
from pathlib import Path


def run_benchmark(
    model_name: str,
    host: str = "127.0.0.1",
    port: int = 8000,
    num_prompts: int = 20,
):
    """Run the vLLM online serving benchmark."""

    Path("results").mkdir(exist_ok=True)

    command = [
        "vllm",
        "bench",
        "serve",
        "--model",
        model_name,
        "--host",
        host,
        "--port",
        str(port),
        "--random-input-len",
        "128",
        "--random-output-len",
        "128",
        "--num-prompts",
        str(num_prompts),
        "--seed",
        "42",
        "--percentile-metrics",
        "ttft,itl",
        "--metric-percentiles",
        "50,99",
        "--result-dir",
        "results",
    ]

    subprocess.run(command, check=True)


if __name__ == "__main__":
    run_benchmark(
        model_name="Qwen/Qwen3-1.7B",
    )