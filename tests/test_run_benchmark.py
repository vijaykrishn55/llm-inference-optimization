import json

import pytest

from scripts.run_benchmark import (
    build_report,
    save_report,
    select_model_id,
)
from src.inference_pipeline.config import PipelineConfig


def test_select_model_id_uses_first_server_model():
    response = {"data": [{"id": "model-a"}, {"id": "model-b"}]}
    assert select_model_id(response) == "model-a"


def test_select_model_id_accepts_override():
    response = {"data": [{"id": "model-a"}]}
    assert select_model_id(response, "model-b") == "model-b"


def test_select_model_id_rejects_empty_list():
    with pytest.raises(ValueError, match="no available models"):
        select_model_id({"data": []})


def test_build_report_includes_settings_and_results():
    config = PipelineConfig()
    report = build_report(
        config,
        "test-model",
        {"mean_ttft_ms": 20.0},
    )

    assert report["settings"]["model"] == "test-model"
    assert report["settings"]["requests"] == config.benchmark_requests
    assert report["results"]["mean_ttft_ms"] == 20.0
    assert report["timestamp_utc"]


def test_save_report_creates_directory_and_valid_json(tmp_path):
    path = tmp_path / "reports" / "benchmark.json"
    report = {"results": {"tokens_per_second": 12.5}}

    saved_path = save_report(report, path)

    assert saved_path.exists()
    assert json.loads(saved_path.read_text(encoding="utf-8")) == report


def test_save_report_rejects_non_json_number(tmp_path):
    with pytest.raises(ValueError):
        save_report({"value": float("nan")}, tmp_path / "report.json")

def test_select_model_id_accepts_list_response():
    response = [{"id": "artifacts/quantized_model"}]
    assert select_model_id(response) == "artifacts/quantized_model"