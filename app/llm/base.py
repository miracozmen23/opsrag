"""Language model interface consumed by the RAG pipeline."""

from typing import Protocol


class LanguageModel(Protocol):
    """Minimal provider-neutral text generation contract."""

    @property
    def provider_name(self) -> str:
        """Return the configured provider identifier."""

        ...

    @property
    def model_name(self) -> str:
        """Return the configured model identifier."""

        ...

    def generate(self, *, instructions: str, input_text: str) -> str:
        """Generate one text response for the supplied instructions and input."""

        ...

