
# LLM Inference Optimization

Implementation of an optimized Large Language Model inference pipeline covering:

- Post-Training Quantization (PTQ)
- AWQ 4-bit quantization
- Quantization quality evaluation
- vLLM inference
- Continuous batching
- PagedAttention
- Speculative decoding
- Latency and throughput benchmarking
- GPU memory profiling

## Model

TinyLlama/TinyLlama-1.1B-Chat-v1.0

## Project Structure

```text
src/
├── quantization/
├── evaluation/
├── inference/
├── benchmarking/
└── utils/

scripts/
configs/
results/
tests/
```
