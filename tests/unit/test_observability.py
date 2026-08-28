"""Optional Langfuse adapter behavior without external requests."""

from contextlib import AbstractContextManager
from typing import Any

import pytest

from app.core.config import Settings
from app.observability import LangfuseObservability, NoOpObservability
from app.observability.langfuse_client import create_observability


class FakeRawObservation:
    def __init__(self, *, fail_update: bool = False) -> None:
        self.fail_update = fail_update
        self.updates: list[dict[str, Any]] = []

    def update(self, **kwargs: Any) -> None:
        if self.fail_update:
            raise RuntimeError("export update failed")
        self.updates.append(kwargs)


class FakeObservationManager(AbstractContextManager[FakeRawObservation]):
    def __init__(self, observation: FakeRawObservation) -> None:
        self.observation = observation
        self.exit_error_type: type[BaseException] | None = None

    def __enter__(self) -> FakeRawObservation:
        return self.observation

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.exit_error_type = exc_type
        return False


class FakeLangfuseClient:
    def __init__(self, *, fail_update: bool = False) -> None:
        self.manager = FakeObservationManager(
            FakeRawObservation(fail_update=fail_update)
        )
        self.observation_kwargs: list[dict[str, Any]] = []
        self.flush_calls = 0
        self.shutdown_calls = 0

    def start_as_current_observation(self, **kwargs: Any):
        self.observation_kwargs.append(kwargs)
        return self.manager

    def flush(self) -> None:
        self.flush_calls += 1

    def shutdown(self) -> None:
        self.shutdown_calls += 1


def test_factory_is_noop_when_tracing_is_disabled() -> None:
    observer = create_observability(Settings(_env_file=None))
    assert isinstance(observer, NoOpObservability)
    assert observer.enabled is False


def test_factory_is_noop_when_enabled_without_credentials() -> None:
    called = False

    def client_factory(**kwargs: Any) -> None:
        nonlocal called
        called = True

    observer = create_observability(
        Settings(_env_file=None, langfuse_enabled=True),
        client_factory=client_factory,
    )

    assert isinstance(observer, NoOpObservability)
    assert called is False


def test_factory_passes_complete_configuration_to_sdk() -> None:
    captured: dict[str, Any] = {}
    client = FakeLangfuseClient()

    def client_factory(**kwargs: Any) -> FakeLangfuseClient:
        captured.update(kwargs)
        return client

    observer = create_observability(
        Settings(
            _env_file=None,
            app_env="test",
            langfuse_enabled=True,
            langfuse_public_key="pk-lf-test",
            langfuse_secret_key="sk-lf-test",
            langfuse_base_url="https://langfuse.example",
            langfuse_sample_rate=0.25,
        ),
        client_factory=client_factory,
    )

    assert isinstance(observer, LangfuseObservability)
    assert captured == {
        "public_key": "pk-lf-test",
        "secret_key": "sk-lf-test",
        "base_url": "https://langfuse.example",
        "environment": "test",
        "sample_rate": 0.25,
        "tracing_enabled": True,
    }


def test_adapter_forwards_observation_and_lifecycle_calls() -> None:
    client = FakeLangfuseClient()
    observer = LangfuseObservability(client)

    with observer.observe(
        name="rag.generate",
        as_type="generation",
        input={"prompt": "context"},
        model="local-model",
        metadata={"provider": "ollama"},
    ) as observation:
        observation.update(output="answer")
    observer.flush()
    observer.shutdown()

    assert client.observation_kwargs == [
        {
            "name": "rag.generate",
            "as_type": "generation",
            "input": {"prompt": "context"},
            "model": "local-model",
            "metadata": {"provider": "ollama"},
        }
    ]
    assert client.manager.observation.updates == [{"output": "answer"}]
    assert client.flush_calls == 1
    assert client.shutdown_calls == 1


def test_adapter_failures_do_not_break_application_work() -> None:
    class FailingStartClient:
        def start_as_current_observation(self, **kwargs: Any) -> None:
            raise ConnectionError("collector unavailable")

    ran = False
    with LangfuseObservability(FailingStartClient()).observe(
        name="opsrag.query"
    ) as observation:
        ran = True
        observation.update(output="still completed")

    update_failing_client = FakeLangfuseClient(fail_update=True)
    with LangfuseObservability(update_failing_client).observe(
        name="opsrag.query"
    ) as observation:
        observation.update(output="still completed")

    assert ran is True


def test_adapter_preserves_application_exceptions() -> None:
    client = FakeLangfuseClient()

    with pytest.raises(ValueError, match="application failed"):
        with LangfuseObservability(client).observe(name="opsrag.query"):
            raise ValueError("application failed")

    assert client.manager.exit_error_type is ValueError
