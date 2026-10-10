import logging
import sys

from src.inference_pipeline.config import PipelineConfig
from src.inference_pipeline.quantization import run_awq_quantization


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    try:
        report = run_awq_quantization(PipelineConfig())

        print("\nAWQ QUANTIZATION COMPLETE")
        print("=========================")
        print(f"Original size: {report['original_size_mb']:.2f} MB")
        print(f"Quantized size: {report['quantized_size_mb']:.2f} MB")
        print(
            f"Size reduction: "
            f"{report['reduction_percent']:.2f}%"
        )
        print(
            f"Quantization time: "
            f"{report['quantization_seconds']:.2f} seconds"
        )
        print(
            f"Peak GPU memory: "
            f"{report['peak_gpu_memory_gib']:.2f} GiB"
        )
        print("Report: reports/quantization_report.json")

        return 0

    except Exception:
        logging.exception("AWQ quantization failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())