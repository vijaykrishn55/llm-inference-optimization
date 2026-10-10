import json
import logging
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)


def build_benchmark_prompt(target_prompt_tokens: int) -> str:
    """Build a repeatable prompt; actual token count depends on the tokenizer."""
    if target_prompt_tokens <= 0:
        raise ValueError("Target prompt tokens must be positive.")

    sentence = (
        "Explain the trade-offs between model quantization, "
        "KV caching, and continuous batching."
    )
    repeats = max(1, (target_prompt_tokens + 11) // 12)

    return " ".join(
        f"Benchmark case {index + 1}: {sentence}"
        for index in range(repeats)
    )


def benchmark_one_request(
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout: float = 120.0,
    api_key: str | None = None,
) -> dict:
    """Send one streaming request and measure its latency."""

    if not base_url.strip():
        raise ValueError("Base URL cannot be empty.")
    if not model.strip():
        raise ValueError("Model cannot be empty.")
    if not prompt.strip():
        raise ValueError("Prompt cannot be empty.")
    if max_tokens <= 0:
        raise ValueError("Maximum output tokens must be positive.")
    if timeout <= 0:
        raise ValueError("Timeout must be positive.")

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    request = Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    start_time = time.perf_counter()
    output_events = []
    first_output_time = None
    usage = None

    try:
        with urlopen(request, timeout=timeout) as response:
            for raw_line in response:
                line = (
                    raw_line.decode("utf-8")
                    if isinstance(raw_line, bytes)
                    else raw_line
                ).strip()

                if not line.startswith("data:"):
                    continue

                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break

                try:
                    event = json.loads(data)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        f"Invalid streaming response: {data}"
                    ) from exc

                if isinstance(event.get("usage"), dict):
                    usage = event["usage"]

                choices = event.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    content = delta.get("content")

                    if isinstance(content, str) and content:
                        event_time = time.perf_counter()
                        output_events.append(event_time)

                        if first_output_time is None:
                            first_output_time = event_time

    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"vLLM returned HTTP {exc.code}: {detail}"
        ) from exc
    except (URLError, TimeoutError) as exc:
        raise RuntimeError(f"Could not reach vLLM: {exc}") from exc

    end_time = time.perf_counter()

    if not output_events or first_output_time is None:
        raise RuntimeError("The server returned no generated text.")

    latency_seconds = end_time - start_time
    ttft_ms = (first_output_time - start_time) * 1000

    intervals_ms = [
        (current - previous) * 1000
        for previous, current in zip(output_events, output_events[1:])
    ]
    itl_ms = statistics.fmean(intervals_ms) if intervals_ms else None

    reported_output_tokens = (
        usage.get("completion_tokens") if usage else None
    )
    reported_prompt_tokens = usage.get("prompt_tokens") if usage else None

    if isinstance(reported_output_tokens, int) and reported_output_tokens >= 0:
        output_tokens = reported_output_tokens
        token_count_source = "server_usage"
    else:
        # This counts streamed events, not actual tokenizer tokens.
        output_tokens = len(output_events)
        token_count_source = "stream_events_approximation"

    return {
        "ttft_ms": round(ttft_ms, 3),
        "itl_ms": round(itl_ms, 3) if itl_ms is not None else None,
        "latency_ms": round(latency_seconds * 1000, 3),
        "output_tokens": output_tokens,
        "output_token_count_source": token_count_source,
        "prompt_tokens": reported_prompt_tokens,
        "tokens_per_second": round(
            output_tokens / latency_seconds, 3
        ),
    }


def run_benchmark(
    base_url: str,
    model: str,
    prompt: str,
    request_count: int,
    concurrency: int,
    max_tokens: int,
    timeout: float = 120.0,
    api_key: str | None = None,
) -> dict:
    """Run concurrent requests and summarize the results."""

    if request_count <= 0 or concurrency <= 0 or max_tokens <= 0:
        raise ValueError("Request count, concurrency, and max tokens must be positive.")

    start_time = time.perf_counter()
    records = []
    errors = []

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        future_map = {
            executor.submit(
                benchmark_one_request,
                base_url,
                model,
                f"Benchmark sample {index + 1}. {prompt}",
                max_tokens,
                timeout,
                api_key,
            ): index + 1
            for index in range(request_count)
        }

        for future in as_completed(future_map):
            index = future_map[future]
            try:
                result = future.result()
                result["request_index"] = index
                records.append(result)
            except Exception as exc:
                logger.exception("Benchmark request %s failed", index)
                errors.append({
                    "request_index": index,
                    "error": str(exc),
                })

    elapsed_seconds = time.perf_counter() - start_time

    if not records:
        raise RuntimeError(f"All benchmark requests failed: {errors}")

    records.sort(key=lambda item: item["request_index"])

    ttft_values = [record["ttft_ms"] for record in records]
    latency_values = [record["latency_ms"] for record in records]
    itl_values = [
        record["itl_ms"]
        for record in records
        if record["itl_ms"] is not None
    ]

    total_output_tokens = sum(
        record["output_tokens"] for record in records
    )

    return {
        "model": model,
        "requests_requested": request_count,
        "requests_succeeded": len(records),
        "requests_failed": len(errors),
        "concurrency": concurrency,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "mean_ttft_ms": round(statistics.fmean(ttft_values), 3),
        "median_ttft_ms": round(statistics.median(ttft_values), 3),
        "mean_itl_ms": round(statistics.fmean(itl_values), 3)
        if itl_values else None,
        "mean_latency_ms": round(statistics.fmean(latency_values), 3),
        "total_output_tokens": total_output_tokens,
        "request_throughput_per_second": round(
            len(records) / elapsed_seconds, 3
        ),
        "token_throughput_per_second": round(
            total_output_tokens / elapsed_seconds, 3
        ),
        "approximate_token_count_requests": sum(
            record["output_token_count_source"]
            != "server_usage"
            for record in records
        ),
        "records": records,
        "errors": errors,
    }