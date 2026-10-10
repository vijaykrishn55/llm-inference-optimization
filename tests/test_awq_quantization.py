import pytest

from src.inference_pipeline.config import PipelineConfig
from src.inference_pipeline.quantization import (
    build_awq_recipe,
    require_cuda,
    validate_awq_config,
)


class FakeAWQModifier:
    pass


class FakeQuantizationModifier:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def test_cuda_is_required():
    with pytest.raises(RuntimeError, match="CUDA-enabled GPU"):
        require_cuda(False)


def test_cuda_validation_passes():
    require_cuda(True)


def test_default_awq_configuration_is_valid():
    validate_awq_config(PipelineConfig())


def test_awq_runner_rejects_gptq_configuration():
    config = PipelineConfig(quantization_method="gptq")

    with pytest.raises(ValueError, match="supports AWQ"):
        validate_awq_config(config)


def test_awq_runner_rejects_unsupported_scheme():
    config = PipelineConfig(quantization_scheme="W8A16")

    with pytest.raises(ValueError, match="W4A16_ASYM"):
        validate_awq_config(config)


def test_build_awq_recipe():
    recipe = build_awq_recipe(
        FakeAWQModifier,
        FakeQuantizationModifier,
    )

    assert len(recipe) == 2
    assert isinstance(recipe[0], FakeAWQModifier)
    assert isinstance(recipe[1], FakeQuantizationModifier)

    assert recipe[1].kwargs == {
        "targets": ["Linear"],
        "scheme": "W4A16_ASYM",
        "ignore": ["lm_head"],
    }