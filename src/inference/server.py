import subprocess


def start_server(
    model_path: str,
    host: str = "0.0.0.0",
    port: int = 8000,
):
    """Start the vLLM OpenAI-compatible server."""

    command = [
        "vllm",
        "serve",
        model_path,
        "--host",
        host,
        "--port",
        str(port),
    ]

    return subprocess.Popen(command)