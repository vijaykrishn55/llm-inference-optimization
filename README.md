
# LLM Inference Optimization

Implementation of an optimized LLM inference pipeline covering:

- AWQ 4-bit post-training quantization
- Perplexity and next-token accuracy evaluation
- vLLM deployment with PagedAttention and continuous batching support
- Speculative decoding with draft acceptance metrics
- TTFT, inter-token latency and throughput benchmarking
- GPU memory profiling and benchmark comparison

## Models

- **Target:** `Qwen/Qwen2.5-1.5B-Instruct`
- **Draft:** `Qwen/Qwen2.5-0.5B-Instruct`

## Results

### Quantization

| Metric               |         Result |
| -------------------- | -------------: |
| Original model size  |    3,098.89 MB |
| Quantized model size |    1,626.01 MB |
| Size reduction       |         47.53% |
| Quantization time    | 116.02 seconds |

### Quality evaluation

| Metric              |    FP16 |     AWQ |
| ------------------- | ------: | ------: |
| Perplexity          | 18.7015 | 20.0439 |
| Next-token accuracy |  43.26% |  41.70% |

Quantization increased perplexity by 7.18% and reduced next-token accuracy by 1.56 percentage points.

### Inference benchmarks

| Metric     |        FP16 |          AWQ | AWQ + Speculative |
| ---------- | ----------: | -----------: | ----------------: |
| Mean TTFT  |   176.73 ms |     81.30 ms |         177.86 ms |
| Mean ITL   |    26.26 ms |      8.44 ms |          62.74 ms |
| Throughput | 36.45 tok/s | 110.72 tok/s |       30.33 tok/s |

AWQ improved token throughput by approximately 203.77% over the FP16 baseline in these runs.

Speculative decoding achieved a 49.67% draft acceptance rate but did not improve throughput in this configuration.

Each benchmark used 10 requests at concurrency 1, and all requests succeeded.

## Memory profiling

The pipeline also records GPU memory usage and estimates KV-cache and activation memory.

During speculative decoding:

- Two Tesla T4 GPUs were detected.
- One GPU used 10,147 MiB of 15,360 MiB.
- The second GPU used 3 MiB.
- Estimated KV cache: 14.0 MiB.
- Estimated activations: 13.25 MiB.

The KV-cache and activation figures are estimates, not a complete breakdown of VRAM usage.

## Running the scripts

Run the commands from the project root in an environment with the required dependencies installed.

```bash
python -m scripts.quantize_awq
python -m scripts.evaluate_perplexity
python -m scripts.test_vllm_api
python -m scripts.run_benchmark --output reports/benchmark_optimized.json
python -m scripts.profile_memory
```

For benchmark comparison, run the comparison script after generating the baseline and optimized reports:

```bash
python -m scripts.compare_benchmark
```

Quantization and inference require a compatible GPU environment. GPU-dependent experiments were run on Kaggle with NVIDIA Tesla T4 GPUs.

## Limitations

- Benchmarks used a small sample of 10 requests at concurrency 1.
- Next-token accuracy uses a simple argmax-based metric.
- Memory component figures are estimates.
- Results depend on the hardware, model, runtime settings and workload.
