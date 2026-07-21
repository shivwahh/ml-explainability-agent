"""
explanation_provider.py

Config-driven explanation providers.

Decouples the explanation node from a hardcoded LLM client. A provider
turns a rendered prompt into explanation text; the concrete provider is
selected from configuration. No provider performs any network or client
initialization at import time.
"""

import os

import certifi
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def _repair_ca_bundle_env() -> None:
    """
    Point stale CA-bundle environment variables at a valid bundle.

    Tools such as httpx (used by the OpenAI client) read ``SSL_CERT_FILE``
    and related variables to locate the trusted CA bundle. When one of
    these points at a path that no longer exists, client construction
    fails with an opaque ``FileNotFoundError: [Errno 2] ...``. Any file
    variable that references a missing path is redirected to certifi's
    bundle; missing directory variables are removed.
    """
    for name in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
        path = os.environ.get(name)
        if path and not os.path.isfile(path):
            os.environ[name] = certifi.where()

    cert_dir = os.environ.get("SSL_CERT_DIR")
    if cert_dir and not os.path.isdir(cert_dir):
        del os.environ["SSL_CERT_DIR"]


_repair_ca_bundle_env()


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


class GeminiExplanationProvider(ExplanationProvider):
    """
    Google Gemini-backed provider.

    The Gemini client is constructed lazily inside :meth:`generate`,
    so importing this module never requires API keys or library availability
    at import time.
    """

    def __init__(self, model: str = "gemini-3.6-flash"):
        self.model = model

    def generate(self, prompt: str) -> str:
        from google import genai

        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if api_key:
            client = genai.Client(api_key=api_key)
        else:
            client = genai.Client()

        response = client.models.generate_content(
            model=self.model,
            contents=prompt,
        )

        return response.text


PROVIDERS = {
    "echo": EchoExplanationProvider,
    "openai": OpenAIExplanationProvider,
    "gemini": GeminiExplanationProvider,
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
    
    default_model = "gpt-4.1-mini"
    if str(name).lower() == "gemini":
        default_model = "gemini-3.6-flash"
        
    model = config.get("explanation", "model", default=default_model)

    provider_cls = PROVIDERS.get(str(name).lower(), EchoExplanationProvider)

    return provider_cls(model=model)
