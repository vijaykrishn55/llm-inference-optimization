import logging
import os
import sys

from src.inference_pipeline.api_client import VLLMClient


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    try:
        client = VLLMClient(
            base_url=os.getenv(
                "VLLM_BASE_URL",
                "http://127.0.0.1:8000",
            ),
            api_key=os.getenv("VLLM_API_KEY"),
            timeout=60.0,
        )

        client.check_health()
        print("✓ vLLM server is healthy")

        models = client.list_models()

        if not models or not models[0].get("id"):
            raise RuntimeError("No available model ID was returned.")

        model_id = models[0]["id"]
        print(f"✓ Model available: {model_id}")

        result = client.chat_completion(
            model=model_id,
            prompt="Explain quantization in two simple sentences.",
            max_tokens=96,
        )

        content = result["choices"][0]["message"].get("content", "")

        print("\nMODEL RESPONSE")
        print("--------------")
        print(content)

        print("\n✓ Chat completion successful")
        return 0

    except Exception:
        logging.exception("vLLM API verification failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())