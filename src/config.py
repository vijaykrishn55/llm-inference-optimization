from pathlib import Path

import yaml


def load_config(path: str = "configs/config.yaml") -> dict:
    """Load project configuration from YAML."""

    config_path = Path(path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)