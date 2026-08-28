"""Optional Langfuse SDK v4 adapter with fail-open behavior."""

import logging
from contextlib import contextmanager
from typing import Any, Callable, Iterator

from app.core.config import Settings
from app.observability.base import (
    NoOpObservability,
    Observation,
    ObservationType,
    ObservabilityClient,
)

logger = logging.getLogger(__name__)


class _SafeObservation:
    """Prevent exporter update errors from affecting request execution."""

    def __init__(self, observation: Any, name: str) -> None:
        self._observation = observation
        self._name = name

    def update(self, **kwargs: Any) -> None:
        try:
            self._observation.update(**kwargs)
        except Exception as exc:
            logger.warning(
                "langfuse_observation_update_failed name=%s error_type=%s",
                self._name,
                type(exc).__name__,
            )


class LangfuseObservability:
    """Adapt the Langfuse client to the application's tracing boundary."""

    def __init__(self, client: Any) -> None:
        self._client = client

    @property
    def enabled(self) -> bool:
        return True

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
        kwargs: dict[str, Any] = {
            "name": name,
            "as_type": as_type,
        }
        if input is not None:
            kwargs["input"] = input
        if model is not None:
            kwargs["model"] = model
        if metadata is not None:
            kwargs["metadata"] = metadata

        try:
            manager = self._client.start_as_current_observation(**kwargs)
            raw_observation = manager.__enter__()
        except Exception as exc:
            logger.warning(
                "langfuse_observation_start_failed name=%s error_type=%s",
                name,
                type(exc).__name__,
            )
            with NoOpObservability().observe(name=name, as_type=as_type) as fallback:
                yield fallback
            return

        observation = _SafeObservation(raw_observation, name)
        try:
            yield observation
        except BaseException as operation_error:
            try:
                manager.__exit__(
                    type(operation_error),
                    operation_error,
                    operation_error.__traceback__,
                )
            except Exception as exporter_error:
                logger.warning(
                    "langfuse_observation_close_failed name=%s error_type=%s",
                    name,
                    type(exporter_error).__name__,
                )
            raise
        else:
            try:
                manager.__exit__(None, None, None)
            except Exception as exc:
                logger.warning(
                    "langfuse_observation_close_failed name=%s error_type=%s",
                    name,
                    type(exc).__name__,
                )

    def flush(self) -> None:
        self._call_safely("flush")

    def shutdown(self) -> None:
        self._call_safely("shutdown")

    def _call_safely(self, method_name: str) -> None:
        try:
            getattr(self._client, method_name)()
        except Exception as exc:
            logger.warning(
                "langfuse_%s_failed error_type=%s",
                method_name,
                type(exc).__name__,
            )


def create_observability(
    settings: Settings,
    *,
    client_factory: Callable[..., Any] | None = None,
) -> ObservabilityClient:
    """Create Langfuse tracing only after explicit, complete configuration."""

    if not settings.langfuse_enabled:
        logger.info("langfuse_disabled")
        return NoOpObservability()

    public_key = settings.langfuse_public_key.strip()
    secret_key = (
        settings.langfuse_secret_key.get_secret_value()
        if settings.langfuse_secret_key is not None
        else ""
    )
    if not public_key or not secret_key:
        logger.warning("langfuse_disabled_missing_credentials")
        return NoOpObservability()

    if client_factory is None:
        try:
            from langfuse import Langfuse
        except ModuleNotFoundError:
            logger.warning("langfuse_disabled_missing_sdk")
            return NoOpObservability()
        client_factory = Langfuse

    try:
        client = client_factory(
            public_key=public_key,
            secret_key=secret_key,
            base_url=settings.langfuse_base_url,
            environment=settings.app_env,
            sample_rate=settings.langfuse_sample_rate,
            tracing_enabled=True,
        )
    except Exception as exc:
        logger.warning(
            "langfuse_disabled_initialization_failed error_type=%s",
            type(exc).__name__,
        )
        return NoOpObservability()

    logger.info(
        "langfuse_enabled base_url=%s environment=%s sample_rate=%.3f",
        settings.langfuse_base_url,
        settings.app_env,
        settings.langfuse_sample_rate,
    )
    return LangfuseObservability(client)
