import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from src.inference_pipeline.api_client import VLLMClient
from src.inference_pipeline.benchmark_runner import (
    build_benchmark_prompt,
    run_benchmark,
)
from src.inference_pipeline.config import PipelineConfig

logger = logging.getLogger(__name__)


def select_model_id(
    models_response: dict | list,
    requested_model: str | None = None,
) -> str:
    """Choose a requested model or the first model served by vLLM."""
    if requested_model:
        return requested_model

    if isinstance(models_response, list):
        models = models_response
    elif isinstance(models_response, dict):
        models = models_response.get("data", [])
    else:
        raise ValueError("Invalid model-list response from vLLM.")

    if not models:
        raise ValueError("The vLLM server returned no available models.")

    model_id = models[0].get("id")
    if not isinstance(model_id, str) or not model_id:
        raise ValueError("The server response has no valid model ID.")

    return model_id


def build_report(
    config: PipelineConfig,
    model: str,
    results: dict,
) -> dict:
    """Combine benchmark results with the settings used."""
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "server": {
            "host": config.server_host,
            "port": config.server_port,
        },
        "settings": {
            "model": model,
            "requests": config.benchmark_requests,
            "concurrency": config.benchmark_concurrency,
            "prompt_tokens_target": config.prompt_tokens,
            "generation_tokens": config.generation_tokens,
        },
        "results": results,
    }


def save_report(report: dict, output_path: str | Path) -> Path:
    """Write the report as formatted JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, allow_nan=False)
        file.write("\n")

    return path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark a running vLLM server."
    )
    parser.add_argument(
        "--model",
        help="Model ID returned by the server; defaults to its first model.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Report path; defaults to reports/benchmark.json.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    config = PipelineConfig()
    base_url = f"http://{config.server_host}:{config.server_port}"
    output_path = args.output or config.benchmark_report_path

    try:
        client = VLLMClient(base_url=base_url)
        model = select_model_id(client.list_models(), args.model)

        logger.info("Benchmarking model: %s", model)

        results = run_benchmark(
            base_url=base_url,
            model=model,
            prompt=build_benchmark_prompt(config.prompt_tokens),
            request_count=config.benchmark_requests,
            concurrency=config.benchmark_concurrency,
            max_tokens=config.generation_tokens,
        )

        report = build_report(config, model, results)
        saved_path = save_report(report, output_path)

    except Exception:
        logger.exception("Benchmark failed.")
        return 1

    print(f"\nBenchmark report saved to: {saved_path}")
    print(f"Requests succeeded: {results['requests_succeeded']}")
    print(f"Requests failed: {results['requests_failed']}")
    print(f"Mean TTFT: {results['mean_ttft_ms']} ms")
    print(f"Mean ITL: {results['mean_itl_ms']} ms")
    print(f"Token throughput: {results['token_throughput_per_second']} tokens/sec")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())