import httpx
import pytest
from openai import OpenAI

import litellm
from litellm.llms.aiml.chat.transformation import AIMLChatConfig

ATTRIBUTION_KEYS = ("http-referer", "x-title", "x-aimlapi-partner-id", "x-aimlapi-source")


def _recording_client(base_url: str, sent: list[httpx.Request]) -> OpenAI:
    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "created": 0,
                "model": "openai/gpt-5-5",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    return OpenAI(
        api_key="sk-test",
        base_url=base_url,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def test_completion_sends_every_attribution_header_on_the_wire(monkeypatch):
    monkeypatch.setenv("AIML_API_KEY", "sk-test")
    sent: list[httpx.Request] = []

    litellm.completion(
        model="aiml/openai/gpt-5-5",
        messages=[{"role": "user", "content": "hi"}],
        client=_recording_client("https://api.aimlapi.com/v1", sent),
    )

    assert len(sent) == 1
    headers = sent[0].headers
    assert headers["http-referer"] == "https://github.com/BerriAI/litellm"
    assert headers["x-title"] == "LiteLLM"
    assert headers["x-aimlapi-partner-id"] == "part_litellm"
    assert headers["x-aimlapi-source"] == "agent/litellm"


def test_completion_keeps_caller_supplied_headers(monkeypatch):
    monkeypatch.setenv("AIML_API_KEY", "sk-test")
    sent: list[httpx.Request] = []

    litellm.completion(
        model="aiml/openai/gpt-5-5",
        messages=[{"role": "user", "content": "hi"}],
        extra_headers={"X-Title": "my-app", "X-Custom": "kept"},
        client=_recording_client("https://api.aimlapi.com/v1", sent),
    )

    headers = sent[0].headers
    assert headers["x-title"] == "my-app"
    assert headers["x-custom"] == "kept"
    assert headers["x-aimlapi-partner-id"] == "part_litellm"


def test_completion_withholds_attribution_from_a_non_aimlapi_base(monkeypatch):
    monkeypatch.setenv("AIML_API_KEY", "sk-test")
    monkeypatch.setenv("AIML_API_BASE", "https://gateway.example.com/v1")
    sent: list[httpx.Request] = []

    litellm.completion(
        model="aiml/openai/gpt-5-5",
        messages=[{"role": "user", "content": "hi"}],
        client=_recording_client("https://gateway.example.com/v1", sent),
    )

    headers = sent[0].headers
    assert [key for key in ATTRIBUTION_KEYS if key in headers] == []


def test_other_providers_never_receive_aimlapi_attribution(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    sent: list[httpx.Request] = []

    litellm.completion(
        model="openai/gpt-5-5",
        messages=[{"role": "user", "content": "hi"}],
        client=_recording_client("https://api.openai.com/v1", sent),
    )

    headers = sent[0].headers
    assert [key for key in ATTRIBUTION_KEYS if key in headers] == []


@pytest.mark.parametrize(
    "env_key, env_value",
    [("AIML_API_KEY", "from-aiml"), ("AIMLAPI_API_KEY", "from-aimlapi")],
)
def test_provider_info_reads_both_api_key_env_vars(monkeypatch, env_key, env_value):
    monkeypatch.delenv("AIML_API_KEY", raising=False)
    monkeypatch.delenv("AIMLAPI_API_KEY", raising=False)
    monkeypatch.delenv("AIMLAPI_KEY", raising=False)
    monkeypatch.setenv(env_key, env_value)

    api_base, api_key = AIMLChatConfig()._get_openai_compatible_provider_info(None, None)

    assert api_base == "https://api.aimlapi.com/v1"
    assert api_key == env_value
