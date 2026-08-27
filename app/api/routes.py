"""HTTP endpoints."""

import logging

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.dependencies import get_rag_pipeline
from app.api.schemas import AskRequest, AskResponse, HealthResponse
from app.rag.pipeline import RAGPipeline, RAGPipelineError

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["system"])
async def health() -> HealthResponse:
    """Report API process health without requiring external services."""

    return HealthResponse()


@router.post("/api/v1/ask", response_model=AskResponse, tags=["rag"])
def ask(
    request: AskRequest,
    pipeline: RAGPipeline = Depends(get_rag_pipeline),
) -> AskResponse:
    """Answer a question from retrieved knowledge-base context."""

    try:
        result = pipeline.answer(request.question)
    except RAGPipelineError as exc:
        logger.warning("rag_request_rejected reason=%s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("rag_request_failed")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The RAG service is temporarily unavailable.",
        ) from exc

    return AskResponse.model_validate(result.model_dump())
