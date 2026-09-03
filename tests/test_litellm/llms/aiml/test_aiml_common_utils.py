import re

import pytest

from litellm.llms.aiml.common_utils import (
    AIML_ATTRIBUTION_HEADERS,
    aiml_attribution_headers,
    get_aiml_api_base,
    get_aiml_api_key,
    with_aiml_attribution,
)

PARTNER_ID_PATTERN = re.compile(r"^part_[A-Za-z0-9]{1,64}$")
SOURCE_PATTERN = re.compile(r"^(web|agent|mcp)/[a-z0-9-]{1,32}$")


def test_partner_id_matches_the_gateway_contract():
    assert PARTNER_ID_PATTERN.match(AIML_ATTRIBUTION_HEADERS["X-AIMLAPI-Partner-ID"])


def test_source_matches_the_gateway_contract():
    assert SOURCE_PATTERN.match(AIML_ATTRIBUTION_HEADERS["X-AIMLAPI-Source"])


def test_referer_and_title_identify_the_calling_project():
    assert AIML_ATTRIBUTION_HEADERS["HTTP-Referer"] == "https://github.com/BerriAI/litellm"
    assert AIML_ATTRIBUTION_HEADERS["X-Title"] == "LiteLLM"


def test_attribution_is_sent_to_the_default_base():
    assert dict(aiml_attribution_headers(None)) == dict(AIML_ATTRIBUTION_HEADERS)


def test_attribution_is_withheld_from_a_proxy_fronting_the_api():
    assert dict(aiml_attribution_headers("https://gateway.example.com/aiml/v1")) == {}


def test_caller_headers_win_on_a_key_clash():
    merged = with_aiml_attribution({"X-Title": "my-app"}, None)

    assert merged["X-Title"] == "my-app"
    assert merged["X-AIMLAPI-Partner-ID"] == AIML_ATTRIBUTION_HEADERS["X-AIMLAPI-Partner-ID"]


def test_merging_never_mutates_the_shared_constant():
    before = dict(AIML_ATTRIBUTION_HEADERS)

    first = with_aiml_attribution({"X-Title": "first"}, None)
    first["X-Request-Id"] = "abc"
    second = with_aiml_attribution(None, None)

    assert dict(AIML_ATTRIBUTION_HEADERS) == before
    assert "X-Request-Id" not in second


def test_api_key_falls_back_across_both_env_var_spellings(monkeypatch):
    monkeypatch.delenv("AIML_API_KEY", raising=False)
    monkeypatch.setenv("AIMLAPI_API_KEY", "from-aimlapi-spelling")

    assert get_aiml_api_key() == "from-aimlapi-spelling"


def test_existing_env_var_still_takes_precedence(monkeypatch):
    monkeypatch.setenv("AIML_API_KEY", "from-aiml-spelling")
    monkeypatch.setenv("AIMLAPI_API_KEY", "from-aimlapi-spelling")

    assert get_aiml_api_key() == "from-aiml-spelling"


def test_explicit_api_key_beats_the_environment(monkeypatch):
    monkeypatch.setenv("AIML_API_KEY", "from-env")

    assert get_aiml_api_key("explicit") == "explicit"


@pytest.mark.parametrize(
    "env_base, expected",
    [
        (None, "https://api.aimlapi.com/v1"),
        ("https://gateway.example.com/v1", "https://gateway.example.com/v1"),
    ],
)
def test_api_base_resolution(monkeypatch, env_base, expected):
    monkeypatch.delenv("AIML_API_BASE", raising=False)
    if env_base is not None:
        monkeypatch.setenv("AIML_API_BASE", env_base)

    assert get_aiml_api_base() == expected
