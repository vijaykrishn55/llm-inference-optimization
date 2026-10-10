import io
import json
from urllib.error import HTTPError, URLError

import pytest

from src.inference_pipeline.api_client import VLLMClient


class FakeResponse:
    def __init__(self, body=b""):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self.body


def test_client_configuration():
    client = VLLMClient("http://localhost:8000/")

    assert client.base_url == "http://localhost:8000"


def test_invalid_url_is_rejected():
    with pytest.raises(ValueError):
        VLLMClient("localhost:8000")


def test_invalid_timeout_is_rejected():
    with pytest.raises(ValueError):
        VLLMClient(timeout=0)


def test_health_check(monkeypatch):
    monkeypatch.setattr(
        "src.inference_pipeline.api_client.urlopen",
        lambda request, timeout: FakeResponse(),
    )

    client = VLLMClient()

    assert client.check_health() is True


def test_list_models(monkeypatch):
    body = json.dumps({
        "data": [
            {"id": "quantized-model"}
        ]
    }).encode("utf-8")

    monkeypatch.setattr(
        "src.inference_pipeline.api_client.urlopen",
        lambda request, timeout: FakeResponse(body),
    )

    client = VLLMClient()

    assert client.list_models() == [{"id": "quantized-model"}]


def test_chat_completion_request(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["method"] = request.method
        captured["payload"] = json.loads(request.data.decode("utf-8"))

        return FakeResponse(json.dumps({
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Quantization reduces numerical precision."
                    }
                }
            ]
        }).encode("utf-8"))

    monkeypatch.setattr(
        "src.inference_pipeline.api_client.urlopen",
        fake_urlopen,
    )

    client = VLLMClient()
    result = client.chat_completion(
        model="quantized-model",
        prompt="Explain quantization.",
    )

    assert captured["url"].endswith("/v1/chat/completions")
    assert captured["method"] == "POST"
    assert captured["payload"]["model"] == "quantized-model"
    assert result["choices"][0]["message"]["role"] == "assistant"


def test_api_key_is_sent_as_bearer_token(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured.update(request.header_items())
        return FakeResponse(b"")

    monkeypatch.setattr(
        "src.inference_pipeline.api_client.urlopen",
        fake_urlopen,
    )

    VLLMClient(api_key="test-key").check_health()

    assert captured["Authorization"] == "Bearer test-key"


def test_http_error_is_reported(monkeypatch):
    def fake_urlopen(request, timeout):
        raise HTTPError(
            request.full_url,
            503,
            "Service Unavailable",
            {},
            io.BytesIO(b"server unavailable"),
        )

    monkeypatch.setattr(
        "src.inference_pipeline.api_client.urlopen",
        fake_urlopen,
    )

    with pytest.raises(RuntimeError, match="HTTP 503"):
        VLLMClient().check_health()


def test_connection_error_is_reported(monkeypatch):
    monkeypatch.setattr(
        "src.inference_pipeline.api_client.urlopen",
        lambda request, timeout: (_ for _ in ()).throw(
            URLError("connection refused")
        ),
    )

    with pytest.raises(RuntimeError, match="Could not connect"):
        VLLMClient().check_health()


def test_empty_prompt_is_rejected():
    with pytest.raises(ValueError):
        VLLMClient().chat_completion(
            model="test",
            prompt=" ",
        )


def test_invalid_model_response_is_rejected(monkeypatch):
    monkeypatch.setattr(
        "src.inference_pipeline.api_client.urlopen",
        lambda request, timeout: FakeResponse(b"not-json"),
    )

    with pytest.raises(RuntimeError, match="invalid JSON"):
        VLLMClient().list_models()