"""Tests for the Streamlit connection settings."""

import pytest
from pydantic import ValidationError

from frontend.config import FrontendSettings


def test_frontend_settings_normalize_api_url() -> None:
    settings = FrontendSettings(
        api_base_url=" https://opsrag.example.test/ ",
        api_timeout_seconds=12,
    )

    assert settings.api_base_url == "https://opsrag.example.test"
    assert settings.api_timeout_seconds == 12


@pytest.mark.parametrize("url", ["", "localhost:8000", "ftp://localhost", "http://"])
def test_frontend_settings_reject_invalid_api_url(url: str) -> None:
    with pytest.raises(ValidationError, match="valid HTTP"):
        FrontendSettings(api_base_url=url)


def test_frontend_settings_reject_non_positive_timeout() -> None:
    with pytest.raises(ValidationError):
        FrontendSettings(api_timeout_seconds=0)
