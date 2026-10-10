import json

import pytest

from src.inference_pipeline import benchmark_runner as runner


class FakeResponse:
    def __init__(self, events):
        self.lines = []
        for event in events:
            self.lines.append(
                f"data: {json.dumps(event)}\n".encode("utf-8")
            )
            self.lines.append(b"\n")
        self.lines.append(b"data: [DONE]\n")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def __iter__(self):
        return iter(self.lines)


def test_build_benchmark_prompt():
    prompt = runner.build_benchmark_prompt(24)
    assert "Benchmark case 1" in prompt
    assert "Benchmark case 2" in prompt


def test_build_benchmark_prompt_rejects_invalid_length():
    with pytest.raises(ValueError):
        runner.build_benchmark_prompt(0)


def test_streaming_request_uses_server_token_usage(monkeypatch):
    events = [
        {"choices": [{"delta": {"content": "Hello"}}]},
        {"choices": [{"delta": {"content": " world"}}]},
        {
            "choices": [],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2},
        },
    ]
    monkeypatch.setattr(
        runner, "urlopen", lambda *args, **kwargs: FakeResponse(events)
    )

    result = runner.benchmark_one_request(
        "http://localhost:8000", "test-model", "Hello", 10
    )

    assert result["ttft_ms"] >= 0
    assert result["latency_ms"] >= result["ttft_ms"]
    assert result["output_tokens"] == 2
    assert result["prompt_tokens"] == 10
    assert result["output_token_count_source"] == "server_usage"
    assert result["itl_ms"] is not None


def test_streaming_request_marks_fallback_as_approximate(monkeypatch):
    events = [
        {"choices": [{"delta": {"content": "Hello"}}]},
        {"choices": [{"delta": {"content": " world"}}]},
    ]
    monkeypatch.setattr(
        runner, "urlopen", lambda *args, **kwargs: FakeResponse(events)
    )

    result = runner.benchmark_one_request(
        "http://localhost:8000", "test-model", "Hello", 10
    )

    assert result["output_tokens"] == 2
    assert result["output_token_count_source"] == "stream_events_approximation"


def test_streaming_request_rejects_empty_output(monkeypatch):
    monkeypatch.setattr(
        runner, "urlopen", lambda *args, **kwargs: FakeResponse([])
    )

    with pytest.raises(RuntimeError, match="no generated text"):
        runner.benchmark_one_request(
            "http://localhost:8000", "test-model", "Hello", 10
        )


def test_run_benchmark_summarizes_results(monkeypatch):
    def fake_request(*args, **kwargs):
        return {
            "ttft_ms": 20.0,
            "itl_ms": 5.0,
            "latency_ms": 100.0,
            "output_tokens": 10,
            "output_token_count_source": "server_usage",
            "prompt_tokens": 12,
            "tokens_per_second": 100.0,
        }

    monkeypatch.setattr(runner, "benchmark_one_request", fake_request)

    result = runner.run_benchmark(
        base_url="http://localhost:8000",
        model="test-model",
        prompt="Test prompt",
        request_count=3,
        concurrency=2,
        max_tokens=20,
    )

    assert result["requests_requested"] == 3
    assert result["requests_succeeded"] == 3
    assert result["requests_failed"] == 0
    assert result["total_output_tokens"] == 30
    assert result["mean_ttft_ms"] == 20.0
    assert len(result["records"]) == 3


def test_run_benchmark_rejects_invalid_request_count():
    with pytest.raises(ValueError):
        runner.run_benchmark(
            "http://localhost:8000",
            "test-model",
            "Test prompt",
            request_count=0,
            concurrency=1,
            max_tokens=10,
        )


def test_run_benchmark_fails_when_all_requests_fail(monkeypatch):
    def failed_request(*args, **kwargs):
        raise RuntimeError("Server unavailable")

    monkeypatch.setattr(runner, "benchmark_one_request", failed_request)

    with pytest.raises(RuntimeError, match="All benchmark requests failed"):
        runner.run_benchmark(
            "http://localhost:8000",
            "test-model",
            "Test prompt",
            request_count=2,
            concurrency=1,
            max_tokens=10,
        )