import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from transformers import AutoConfig

from src.inference_pipeline.config import PipelineConfig
from src.inference_pipeline.profiling import (
    build_memory_profile,
    collect_gpu_stats,
    fetch_vllm_metrics,
)

logger = logging.getLogger(__name__)


def get_config_value(config, *names, default=None):
    for name in names:
        value = getattr(config, name, None)
        if value is not None:
            return value

    if default is not None:
        return default

    raise ValueError(f"Could not find model configuration field: {names}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Profile GPU and model memory usage."
    )
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--sequence-length", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    pipeline = PipelineConfig()
    model_dir = args.model_dir or pipeline.quantized_model_dir
    batch_size = args.batch_size or pipeline.benchmark_concurrency
    sequence_length = args.sequence_length or pipeline.max_sequence_length
    output_path = args.output or pipeline.report_dir / "memory_profile.json"
    base_url = f"http://{pipeline.server_host}:{pipeline.server_port}"

    try:
        model_config = AutoConfig.from_pretrained(str(model_dir))

        num_layers = get_config_value(
            model_config, "num_hidden_layers", "n_layer"
        )
        hidden_size = get_config_value(
            model_config, "hidden_size", "n_embd"
        )
        intermediate_size = get_config_value(
            model_config,
            "intermediate_size",
            "n_inner",
            default=4 * hidden_size,
        )
        num_attention_heads = get_config_value(
            model_config, "num_attention_heads", "n_head"
        )
        num_kv_heads = get_config_value(
            model_config,
            "num_key_value_heads",
            default=num_attention_heads,
        )
        head_dim = get_config_value(
            model_config,
            "head_dim",
            default=hidden_size // num_attention_heads,
        )

        gpu_stats = collect_gpu_stats()

        try:
            vllm_metrics = fetch_vllm_metrics(base_url)
        except RuntimeError as exc:
            logger.warning("Could not collect vLLM metrics: %s", exc)
            vllm_metrics = None

        profile = build_memory_profile(
            model_dir=model_dir,
            gpu_stats=gpu_stats,
            num_layers=num_layers,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            batch_size=batch_size,
            sequence_length=sequence_length,
            bytes_per_element=2,
            vllm_metrics=vllm_metrics,
        )

        report = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "model_dir": str(model_dir),
            "profile": profile,
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    except Exception:
        logger.exception("Memory profiling failed.")
        return 1

    print(f"Memory profile saved to: {output_path}")
    print(f"GPU(s) detected: {len(gpu_stats)}")
    for gpu in gpu_stats:
        print(
            f"{gpu['gpu_name']}: "
            f"{gpu['memory_used_mib']:.0f} MiB used / "
            f"{gpu['memory_total_mib']:.0f} MiB total"
        )

    print(
        "Estimated KV cache: "
        f"{profile['memory_components']['kv_cache_estimate_bytes']:,} bytes"
    )
    print(
        "Estimated activations: "
        f"{profile['memory_components']['activation_estimate_bytes']:,} bytes"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())