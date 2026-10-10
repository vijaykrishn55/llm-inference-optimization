import pytest

from src.inference_pipeline.inference_schema import (
    InferenceRecord,
    summarize_inference_records,
)


def create_record(
    request_id="1",
    model_variant="baseline",
    ttft_ms=200.0,
    latency_ms=1000.0,
    output_tokens=40,
    itl_ms=20.0,
):
    return InferenceRecord(
        request_id=request_id,
        model_variant=model_variant,
        prompt_tokens=100,
        output_tokens=output_tokens,
        ttft_ms=ttft_ms,
        latency_ms=latency_ms,
        itl_ms=itl_ms,
    )


def test_inference_record_creation():
    record = create_record()

    assert record.model_variant == "baseline"
    assert record.prompt_tokens == 100
    assert record.output_tokens == 40


def test_record_serialization():
    record = create_record()

    result = record.to_dict()

    assert result["request_id"] == "1"
    assert result["ttft_ms"] == 200.0


def test_ttft_cannot_exceed_latency():
    with pytest.raises(ValueError):
        create_record(ttft_ms=1200.0, latency_ms=1000.0)


def test_invalid_token_count_is_rejected():
    with pytest.raises(ValueError):
        create_record(output_tokens=0)


def test_benchmark_summary():
    records = [
        create_record(
            request_id="1",
            ttft_ms=200.0,
            latency_ms=1000.0,
            output_tokens=40,
            itl_ms=20.0,
        ),
        create_record(
            request_id="2",
            ttft_ms=300.0,
            latency_ms=1200.0,
            output_tokens=60,
            itl_ms=30.0,
        ),
    ]

    summary = summarize_inference_records(
        records,
        elapsed_seconds=5.0,
    )

    assert summary["request_count"] == 2
    assert summary["total_output_tokens"] == 100
    assert summary["mean_ttft_ms"] == pytest.approx(250.0)
    assert summary["mean_itl_ms"] == pytest.approx(25.0)
    assert summary["tokens_per_second"] == pytest.approx(20.0)
    assert summary["requests_per_second"] == pytest.approx(0.4)


def test_empty_records_are_rejected():
    with pytest.raises(ValueError):
        summarize_inference_records([], elapsed_seconds=5.0)


def test_invalid_benchmark_duration_is_rejected():
    with pytest.raises(ValueError):
        summarize_inference_records(
            [create_record()],
            elapsed_seconds=0,
        )