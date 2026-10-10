import json
import math
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class VLLMClient:
    """Minimal HTTP client for a vLLM OpenAI-compatible server."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000",
        api_key: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        if not base_url.startswith(("http://", "https://")):
            raise ValueError("Base URL must start with http:// or https://")

        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Timeout must be positive and finite.")

        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def _request(
        self,
        path: str,
        method: str = "GET",
        payload: dict | None = None,
    ) -> bytes:
        """Send an HTTP request and return the response body."""

        headers = {"Accept": "application/json"}
        body = None

        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"

        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        request = Request(
            url=f"{self.base_url}{path}",
            data=body,
            headers=headers,
            method=method,
        )

        try:
            with urlopen(request, timeout=self.timeout) as response:
                return response.read()

        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"vLLM returned HTTP {exc.code}: {detail}"
            ) from exc

        except URLError as exc:
            raise RuntimeError(
                f"Could not connect to vLLM: {exc.reason}"
            ) from exc

        except TimeoutError as exc:
            raise RuntimeError("Request to vLLM timed out.") from exc

    @staticmethod
    def _decode_json(body: bytes) -> dict:
        """Decode a JSON response."""

        try:
            result = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("vLLM returned an invalid JSON response.") from exc

        if not isinstance(result, dict):
            raise RuntimeError("Expected a JSON object from vLLM.")

        return result

    def check_health(self) -> bool:
        """Check whether the server health endpoint responds successfully."""

        self._request("/health")
        return True

    def list_models(self) -> list[dict]:
        """Return the models exposed by the server."""

        response = self._decode_json(self._request("/v1/models"))
        models = response.get("data")

        if not isinstance(models, list):
            raise RuntimeError("vLLM response did not contain a models list.")

        return models

    def chat_completion(
        self,
        model: str,
        prompt: str,
        max_tokens: int = 96,
        temperature: float = 0.0,
    ) -> dict:
        """Send a chat completion request and return its JSON response."""

        if not model.strip():
            raise ValueError("Model name cannot be empty.")

        if not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive.")

        if not math.isfinite(temperature) or temperature < 0:
            raise ValueError("Temperature must be finite and non-negative.")

        payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        response = self._decode_json(
            self._request(
                "/v1/chat/completions",
                method="POST",
                payload=payload,
            )
        )

        if not response.get("choices"):
            raise RuntimeError("vLLM returned no completion choices.")

        return response