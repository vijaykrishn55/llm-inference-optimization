import json
import logging
import time
from typing import Any

from src.inference_pipeline.calibration_data import (
    load_wikitext_calibration_texts,
    tokenize_calibration_texts,
)
from src.inference_pipeline.config import PipelineConfig
from src.inference_pipeline.size_metrics import (
    bytes_to_mb,
    compare_model_sizes,
    measure_directory_size,
)

logger = logging.getLogger(__name__)


def require_cuda(cuda_available: bool) -> None:
    """Reject GPU quantization when CUDA is unavailable."""

    if not cuda_available:
        raise RuntimeError(
            "AWQ quantization requires a CUDA-enabled GPU. "
            "Run this step in the Kaggle GPU environment."
        )


def validate_awq_config(config: PipelineConfig) -> None:
    """Validate settings supported by this AWQ runner."""

    if config.quantization_method != "awq":
        raise ValueError("This runner supports AWQ, not GPTQ.")

    if config.quantization_scheme != "W4A16_ASYM":
        raise ValueError(
            "This AWQ runner requires W4A16_ASYM."
        )

    if config.calibration_dataset.lower() != "wikitext":
        raise ValueError(
            "This runner currently supports the WikiText dataset."
        )


def build_awq_recipe(
    awq_modifier_cls: Any,
    quantization_modifier_cls: Any,
    scheme: str = "W4A16_ASYM",
) -> list:
    """Build the AWQ calibration and quantization recipe."""

    try:
        awq_modifier = awq_modifier_cls(duo_scaling="both")
    except TypeError:
        awq_modifier = awq_modifier_cls()

    return [
        awq_modifier,
        quantization_modifier_cls(
            targets=["Linear"],
            scheme=scheme,
            ignore=["lm_head"],
        ),
    ]


def run_awq_quantization(
    config: PipelineConfig,
) -> dict:
    """Load an FP16 model, quantize it with AWQ, and save the results."""

    validate_awq_config(config)

    try:
        import torch
        from datasets import Dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from llmcompressor import oneshot
        from llmcompressor.modifiers.transform.awq import AWQModifier
        from llmcompressor.modifiers.quantization import QuantizationModifier
    except ImportError as exc:
        raise RuntimeError(
            "Missing quantization dependencies. Install the compatible "
            "GPU dependencies in the Kaggle environment."
        ) from exc

    require_cuda(torch.cuda.is_available())

    config.fp16_model_dir.mkdir(parents=True, exist_ok=True)
    config.quantized_model_dir.mkdir(parents=True, exist_ok=True)
    config.report_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading tokenizer: %s", config.target_model_id)

    tokenizer = AutoTokenizer.from_pretrained(
        config.target_model_id,
    )

    logger.info("Loading model in FP16.")

    model = AutoModelForCausalLM.from_pretrained(
        config.target_model_id,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    # Keep an FP16 reference for later evaluation and comparison.
    model.save_pretrained(str(config.fp16_model_dir))
    tokenizer.save_pretrained(str(config.fp16_model_dir))

    original_bytes = measure_directory_size(config.fp16_model_dir)

    logger.info("Preparing WikiText calibration samples.")

    texts = load_wikitext_calibration_texts(
        config.calibration_samples,
    )

    tokenized_records = tokenize_calibration_texts(
        texts=texts,
        tokenizer=tokenizer,
        max_seq_length=config.max_sequence_length,
    )

    calibration_dataset = Dataset.from_list(tokenized_records)

    recipe = build_awq_recipe(
        AWQModifier,
        QuantizationModifier,
        scheme=config.quantization_scheme,
    )

    logger.info(
        "Starting AWQ: model=%s, scheme=%s, samples=%d",
        config.target_model_id,
        config.quantization_scheme,
        len(tokenized_records),
    )

    torch.cuda.reset_peak_memory_stats()
    start_time = time.perf_counter()

    oneshot(
        model=model,
        dataset=calibration_dataset,
        recipe=recipe,
        max_seq_length=config.max_sequence_length,
        num_calibration_samples=len(tokenized_records),
    )

    quantization_seconds = time.perf_counter() - start_time
    peak_gpu_memory_bytes = torch.cuda.max_memory_allocated()

    logger.info("Saving compressed model.")

    model.save_pretrained(
        str(config.quantized_model_dir),
        save_compressed=True,
    )
    tokenizer.save_pretrained(str(config.quantized_model_dir))

    size_comparison = compare_model_sizes(
        config.fp16_model_dir,
        config.quantized_model_dir,
    )

    report = {
        "model_id": config.target_model_id,
        "quantization_method": config.quantization_method,
        "quantization_scheme": config.quantization_scheme,
        "calibration_dataset": config.calibration_dataset,
        "calibration_samples": len(tokenized_records),
        "max_sequence_length": config.max_sequence_length,
        "fp16_model_path": str(config.fp16_model_dir),
        "quantized_model_path": str(config.quantized_model_dir),
        **size_comparison,
        "original_size_mb": bytes_to_mb(
            size_comparison["original_bytes"]
        ),
        "quantized_size_mb": bytes_to_mb(
            size_comparison["quantized_bytes"]
        ),
        "quantization_seconds": quantization_seconds,
        "peak_gpu_memory_bytes": peak_gpu_memory_bytes,
        "peak_gpu_memory_gib": peak_gpu_memory_bytes / (1024 ** 3),
    }

    report_path = config.report_dir / "quantization_report.json"

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    logger.info("Quantization report saved to %s", report_path)

    return report