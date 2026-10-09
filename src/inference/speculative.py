def create_speculative_engine(
    target_model: str,
    draft_model: str,
    num_speculative_tokens: int = 5,
):
    """
    Create a vLLM engine using draft-model speculative decoding.
    """

    from vllm import LLM

    llm = LLM(
        model=target_model,
        speculative_config={
            "method": "draft_model",
            "model": draft_model,
            "num_speculative_tokens": num_speculative_tokens,
        },
    )

    return llm


def generate(
    llm,
    prompt: str,
    max_tokens: int = 128,
):
    """Generate text using the speculative decoding engine."""

    from vllm import SamplingParams

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
    )

    outputs = llm.generate(
        [prompt],
        sampling_params,
    )

    return outputs[0].outputs[0].text