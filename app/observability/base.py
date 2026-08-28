"""Provider-neutral, fail-open observability contracts."""

from contextlib import contextmanager
from typing import Any, ContextManager, Iterator, Literal, Protocol

ObservationType = Literal["span", "chain", "retriever", "generation"]


class Observation(Protocol):
    """One trace observation that can be enriched while work executes."""

    def update(self, **kwargs: Any) -> None:
        """Attach output, metadata, or an error state to the observation."""

        ...


class ObservabilityClient(Protocol):
    """Small tracing boundary used by the application workflow."""

    @property
    def enabled(self) -> bool:
        """Return whether observations are exported."""

        ...

    def observe(
        self,
        *,
        name: str,
        as_type: ObservationType = "span",
        input: object | None = None,
        model: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> ContextManager[Observation]:
        """Create a scoped observation around application work."""

        ...

    def flush(self) -> None:
        """Export queued observations without raising into the application."""

        ...

    def shutdown(self) -> None:
        """Flush and stop background exporter resources."""

        ...


class NoOpObservation:
    """Observation implementation used when tracing is disabled."""

    def update(self, **kwargs: Any) -> None:
        return None


class NoOpObservability:
    """Zero-dependency observability client that performs no work."""

    @property
    def enabled(self) -> bool:
        return False

    @contextmanager
    def observe(
        self,
        *,
        name: str,
        as_type: ObservationType = "span",
        input: object | None = None,
        model: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> Iterator[Observation]:
        yield NoOpObservation()

    def flush(self) -> None:
        return None

    def shutdown(self) -> None:
        return None
