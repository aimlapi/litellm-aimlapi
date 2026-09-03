import types
from collections.abc import Mapping
from typing import Final
from urllib.parse import urlparse

from litellm.secret_managers.main import get_secret_str

AIML_DEFAULT_API_BASE: Final = "https://api.aimlapi.com/v1"

AIML_ATTRIBUTION_HOSTS: Final = frozenset({"api.aimlapi.com"})

AIML_ATTRIBUTION_HEADERS: Final[Mapping[str, str]] = types.MappingProxyType(
    {
        "HTTP-Referer": "https://github.com/BerriAI/litellm",
        "X-Title": "LiteLLM",
        "X-AIMLAPI-Partner-ID": "part_litellm",
        "X-AIMLAPI-Source": "agent/litellm",
    }
)

_NO_ATTRIBUTION: Final[Mapping[str, str]] = types.MappingProxyType({})


def get_aiml_api_key(api_key: str | None = None) -> str | None:
    return (
        api_key or get_secret_str("AIML_API_KEY") or get_secret_str("AIMLAPI_API_KEY") or get_secret_str("AIMLAPI_KEY")
    )


def get_aiml_api_base(api_base: str | None = None) -> str:
    return api_base or get_secret_str("AIML_API_BASE") or AIML_DEFAULT_API_BASE


def aiml_attribution_headers(api_base: str | None) -> Mapping[str, str]:
    host: Final = urlparse(get_aiml_api_base(api_base)).hostname
    if host in AIML_ATTRIBUTION_HOSTS:
        return AIML_ATTRIBUTION_HEADERS
    return _NO_ATTRIBUTION


def with_aiml_attribution(
    headers: Mapping[str, str] | None, api_base: str | None
) -> dict[str, str]:  # mutable-ok: the OpenAI SDK extra_headers slot this feeds is typed as a plain dict
    return {  # mutable-ok: a fresh dict per request leaves the caller's headers and AIML_ATTRIBUTION_HEADERS unmutated
        **aiml_attribution_headers(api_base),
        **(headers or _NO_ATTRIBUTION),
    }
