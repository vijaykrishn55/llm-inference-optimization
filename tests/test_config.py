from src.config import load_config


def test_config_loads():

    config = load_config()

    assert "model" in config
    assert "quantization" in config
    assert "evaluation" in config
    assert "inference" in config
    assert "benchmark" in config