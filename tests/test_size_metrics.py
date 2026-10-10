import pytest

from src.inference_pipeline.size_metrics import (
    bytes_to_mb,
    compare_model_sizes,
    measure_directory_size,
)


def test_measure_directory_size(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    (model_dir / "weights.bin").write_bytes(b"a" * 1000)

    nested_dir = model_dir / "config"
    nested_dir.mkdir()
    (nested_dir / "config.json").write_bytes(b"b" * 500)

    assert measure_directory_size(model_dir) == 1500


def test_compare_model_sizes(tmp_path):
    original = tmp_path / "original"
    quantized = tmp_path / "quantized"

    original.mkdir()
    quantized.mkdir()

    (original / "weights.bin").write_bytes(b"a" * 1000)
    (quantized / "weights.bin").write_bytes(b"b" * 600)

    result = compare_model_sizes(original, quantized)

    assert result["original_bytes"] == 1000
    assert result["quantized_bytes"] == 600
    assert result["reduction_bytes"] == 400
    assert result["reduction_percent"] == pytest.approx(40.0)


def test_missing_directory_raises_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        measure_directory_size(tmp_path / "missing")


def test_empty_original_model_raises_error(tmp_path):
    original = tmp_path / "original"
    quantized = tmp_path / "quantized"

    original.mkdir()
    quantized.mkdir()
    (quantized / "weights.bin").write_bytes(b"data")

    with pytest.raises(ValueError):
        compare_model_sizes(original, quantized)


def test_bytes_to_mb():
    assert bytes_to_mb(1_000_000) == 1.0