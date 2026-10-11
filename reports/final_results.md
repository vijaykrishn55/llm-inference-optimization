# LLM Inference Optimization: Results

## Quantization

| Metric | Result |
|---|---:|
| Method | AWQ, 4-bit W4A16 |
| Original model size | 3,098.89 MB |
| Quantized model size | 1,626.01 MB |
| Size reduction | 47.53% |
| Quantization time | 116.02 seconds |

## Quality

| Metric | FP16 | AWQ |
|---|---:|---:|
| Perplexity (lower is better) | 18.7015 | 20.0439 |
| Next-token accuracy | 43.26% | 41.70% |

Perplexity increased by 7.18%, and next-token accuracy dropped by
1.56 percentage points after quantization.

## Inference benchmarks

| Metric | FP16 baseline | AWQ | AWQ + speculative |
|---|---:|---:|---:|
| Mean TTFT | 176.732 ms | 81.303 ms | 177.862 ms |
| Mean ITL | 26.258 ms | 8.438 ms | 62.738 ms |
| Token throughput | 36.448 tokens/s | 110.717 tokens/s | 30.328 tokens/s |

Each benchmark used 10 requests at concurrency 1. All requests succeeded.

AWQ improved token throughput by about 203.77% over the FP16 baseline
in these runs. Speculative decoding had a 49.67% draft acceptance rate,
but lowered throughput to 30.328 tokens/s. It was therefore not a
performance improvement for this configuration.

## GPU memory profile

- Hardware: 2 Tesla T4 GPUs detected.
- GPU 0: 10,147 MiB used out of 15,360 MiB.
- GPU 1: 3 MiB used out of 15,360 MiB.
- Estimated KV cache: 14.0 MiB.
- Estimated activations: 13.25 MiB.

The component-level memory figures are estimates; they do not account
for the entire VRAM allocation.

## Limitations

- The benchmark sample is small and uses concurrency 1.
- Next-token accuracy is a simple argmax accuracy metric.
- Results depend on model, hardware, runtime configuration, and workload.
