"""FastAPI application entry point."""

from fastapi import FastAPI

from app.api.routes import router
from app.core.config import Settings, get_settings
from app.core.logging import configure_logging


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure an OpsRAG API instance."""

    resolved_settings = settings or get_settings()
    configure_logging(resolved_settings.log_level)
    application = FastAPI(
        title=resolved_settings.app_name,
        version="0.16.0",
        description="Source-grounded technical knowledge assistant.",
    )
    application.include_router(router)
    return application


app = create_app()
