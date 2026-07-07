"""
explanation_provider.py

Config-driven explanation providers.

Decouples the explanation node from a hardcoded LLM client. A provider
turns a rendered prompt into explanation text; the concrete provider is
selected from configuration. No provider performs any network or client
initialization at import time.
"""

from dotenv import load_dotenv
from openai import OpenAI
load_dotenv()


class ExplanationProvider:
    """Base interface for explanation text generation."""

    def generate(self, prompt: str) -> str:
        """
        Return explanation text for the given prompt.

        Args:
            prompt: The fully rendered explanation prompt.

        Returns:
            The generated explanation text.
        """
        raise NotImplementedError


class EchoExplanationProvider(ExplanationProvider):
    """
    Deterministic provider that echoes the prompt.

    Used as the default and for tests so the pipeline runs without any
    API key or network access.
    """

    def __init__(self, model: str = None):
        self.model = model

    def generate(self, prompt: str) -> str:
        return "Explanation (echo provider):\n" + prompt


class OpenAIExplanationProvider(ExplanationProvider):
    """
    OpenAI-backed provider.

    The OpenAI client is imported and constructed lazily inside
    :meth:`generate`, so importing this module never requires an API key.
    """

    def __init__(self, model: str = "gpt-4.1-mini"):
        self.model = model

    def generate(self, prompt: str) -> str:
        

        

        client = OpenAI()

        response = client.responses.create(
            model=self.model,
            input=prompt,
        )

        return response.output_text


PROVIDERS = {
    "echo": EchoExplanationProvider,
    "openai": OpenAIExplanationProvider,
}


def build_provider(config) -> ExplanationProvider:
    """
    Build an explanation provider from configuration.

    Reads ``explanation.provider`` and ``explanation.model``. An unknown
    provider name falls back to the echo provider.

    Args:
        config: A :class:`ConfigLoader` instance.

    Returns:
        A configured :class:`ExplanationProvider`.
    """
    name = config.get("explanation", "provider", default="echo")
    model = config.get("explanation", "model", default="gpt-4.1-mini")

    provider_cls = PROVIDERS.get(str(name).lower(), EchoExplanationProvider)

    return provider_cls(model=model)
