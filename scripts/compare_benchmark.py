import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from src.inference_pipeline.comparison import compare_benchmark_reports

logger = logging.getLogger(__name__)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")

    return data


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare baseline and optimized benchmark reports."
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("reports/benchmark_baseline.json"),
    )
    parser.add_argument(
        "--optimized",
        type=Path,
        default=Path("reports/benchmark_optimized.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/benchmark_comparison.json"),
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        baseline = load_json(args.baseline)
        optimized = load_json(args.optimized)

        report = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "baseline_report": str(args.baseline),
            "optimized_report": str(args.optimized),
            **compare_benchmark_reports(baseline, optimized),
        }

        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    except (OSError, json.JSONDecodeError, ValueError) as exc:
        logger.error("Could not create comparison report: %s", exc)
        return 1

    print(f"Comparison saved to: {args.output}")

    for metric, values in report["metric_comparisons"].items():
        improvement = values["improvement_percent"]
        label = (
            f"{improvement:+.2f}%"
            if improvement is not None
            else "N/A"
        )
        print(f"{metric}: {label} improvement")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())