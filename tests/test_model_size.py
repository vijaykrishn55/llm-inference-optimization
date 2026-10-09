from src.quantization.model_size import calculate_reduction


def test_size_reduction():

    result = calculate_reduction(
        original_bytes=1000,
        quantized_bytes=250,
    )

    assert result["reduction_percent"] == 75.0
    assert result["compression_ratio"] == 4.0