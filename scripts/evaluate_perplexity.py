import gc
import json
import logging
import sys
from pathlib import Path

from src.inference_pipeline.calibration_data import (
    load_wikitext_evaluation_texts,
)
from src.inference_pipeline.config import PipelineConfig
from src.inference_pipeline.evaluation import (
    build_evaluation_report,
    evaluate_model_perplexity,
)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    try:
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
        )

        if not torch.cuda.is_available():
            raise RuntimeError(
                "Model evaluation requires a CUDA-enabled GPU. "
                "Run this script on Kaggle."
            )

        config = PipelineConfig()

        texts = load_wikitext_evaluation_texts(
            config.calibration_samples
        )

        logging.info(
            "Evaluating %d held-out WikiText examples.",
            len(texts),
        )

        # Evaluate the FP16 baseline.
        logging.info("Loading FP16 baseline model.")

        baseline_tokenizer = AutoTokenizer.from_pretrained(
            str(config.fp16_model_dir)
        )

        model = AutoModelForCausalLM.from_pretrained(
            str(config.fp16_model_dir),
            torch_dtype=torch.float16,
            device_map="auto",
        )

        baseline = evaluate_model_perplexity(
            model,
            baseline_tokenizer,
            texts,
            max_seq_length=config.max_sequence_length,
        )

        del model, baseline_tokenizer
        gc.collect()
        torch.cuda.empty_cache()

        # Evaluate the AWQ-compressed model.
        logging.info("Loading AWQ-compressed model.")

        quantized_tokenizer = AutoTokenizer.from_pretrained(
            str(config.quantized_model_dir)
        )

        model = AutoModelForCausalLM.from_pretrained(
            str(config.quantized_model_dir),
            torch_dtype=torch.float16,
            device_map="auto",
        )

        quantized = evaluate_model_perplexity(
            model,
            quantized_tokenizer,
            texts,
            max_seq_length=config.max_sequence_length,
        )

        del model, quantized_tokenizer
        gc.collect()
        torch.cuda.empty_cache()

        report = {
            "model_id": config.target_model_id,
            "evaluation_dataset": "WikiText-2 test split",
            "evaluation_examples_requested": len(texts),
            "max_sequence_length": config.max_sequence_length,
            **build_evaluation_report(baseline, quantized),
        }

        config.report_dir.mkdir(parents=True, exist_ok=True)

        report_path = config.report_dir / "quantization_quality_report.json"

        with report_path.open("w", encoding="utf-8") as file:
            json.dump(report, file, indent=2)

        print("\nQUANTIZATION QUALITY RESULTS")
        print("============================")
        print(f"FP16 perplexity: {baseline['perplexity']:.4f}")
        print(f"AWQ perplexity: {quantized['perplexity']:.4f}")
        print(
            "Perplexity change: "
            f"{report['perplexity_comparison']['change_percent']:.2f}%"
        )
        print(f"FP16 next-token accuracy: {baseline['accuracy']:.4f}")
        print(f"AWQ next-token accuracy: {quantized['accuracy']:.4f}")
        print(
            "Accuracy drop: "
            f"{report['accuracy_comparison']['drop_percentage_points']:.2f} "
            "percentage points"
        )
        print(f"\nReport saved to: {report_path}")

        return 0

    except Exception:
        logging.exception("Perplexity evaluation failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())