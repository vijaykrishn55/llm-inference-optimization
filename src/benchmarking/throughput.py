def tokens_per_second(
    num_tokens: int,
    elapsed_seconds: float,
) -> float:

    if elapsed_seconds <= 0:
        return 0.0

    return num_tokens / elapsed_seconds