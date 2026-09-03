from typing import Final

from litellm.llms.aiml.common_utils import get_aiml_api_base, get_aiml_api_key
from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig


class AIMLChatConfig(OpenAIGPTConfig):
    @property
    def custom_llm_provider(self) -> str | None:
        return "aiml"

    def _get_openai_compatible_provider_info(
        self, api_base: str | None, api_key: str | None
    ) -> tuple[str | None, str | None]:
        resolved_api_base: Final = get_aiml_api_base(api_base)
        dynamic_api_key: Final = get_aiml_api_key(api_key)
        return resolved_api_base, dynamic_api_key
