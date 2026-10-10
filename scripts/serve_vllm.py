import logging
import sys

from src.inference_pipeline.config import PipelineConfig
from src.inference_pipeline.deployment import launch_vllm_server


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    try:
        config = PipelineConfig()

        logging.info(
            "Starting vLLM server for %s",
            config.quantized_model_dir,
        )
        logging.info("Host: %s", config.server_host)
        logging.info("Port: %s", config.server_port)
        logging.info("Maximum sequences: %d", config.max_num_seqs)

        launch_vllm_server(config)
        return 0

    except KeyboardInterrupt:
        logging.info("Server stopped.")
        return 0

    except Exception:
        logging.exception("Failed to launch vLLM.")
        return 1


if __name__ == "__main__":
    sys.exit(main())