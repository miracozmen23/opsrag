"""Minimal Streamlit demo for the OpsRAG FastAPI service."""

import streamlit as st

from app.api.schemas import AskResponse, SourceResponse
from frontend.client import APIClientError, OpsRAGAPIClient
from frontend.config import get_frontend_settings


@st.cache_resource
def get_api_client() -> OpsRAGAPIClient:
    """Reuse the HTTP connection pool across Streamlit reruns."""

    settings = get_frontend_settings()
    return OpsRAGAPIClient(
        base_url=settings.api_base_url,
        timeout_seconds=settings.api_timeout_seconds,
    )


def render_source(source: SourceResponse) -> None:
    """Render one API-owned source record without trusting model metadata."""

    with st.expander(f"[{source.source_id}] {source.title}"):
        st.markdown(f"**Document:** `{source.document}`")
        st.markdown(f"**Section:** {source.section}")
        if source.page_number is not None:
            st.markdown(f"**Page:** {source.page_number}")
        st.markdown(f"**Relevance score:** {source.score:.1%}")
        st.caption(f"Chunks: {', '.join(source.chunk_ids)}")


def render_response(result: AskResponse) -> None:
    """Render the answer, confidence, execution metadata, and sources."""

    st.subheader("Answer")
    st.markdown(result.answer)

    confidence, route, contexts = st.columns(3)
    confidence.metric("Retrieval confidence", f"{result.retrieval_confidence:.1%}")
    route.metric("Route", result.metadata.route.title())
    contexts.metric("Retrieved contexts", result.metadata.retrieved_chunks)

    st.caption(
        "Retrieval: "
        f"{result.metadata.retrieval_method.replace('_', ' ')} · "
        f"Cited sources: {result.metadata.cited_sources}"
    )

    st.subheader("Sources")
    if not result.sources:
        st.info("No knowledge-base sources were used for this response.")
        return
    for source in result.sources:
        render_source(source)


def main() -> None:
    """Render the single-page demo and execute questions on form submission."""

    st.set_page_config(page_title="OpsRAG", page_icon="🔎", layout="centered")
    st.title("OpsRAG")
    st.caption("Source-grounded answers for technical operations questions.")

    with st.form("ask_form"):
        question = st.text_area(
            "Technical question",
            max_chars=4000,
            placeholder="Why does PostgreSQL return connection refused in Docker Compose?",
        )
        submitted = st.form_submit_button(
            "Ask OpsRAG",
            type="primary",
            width="stretch",
        )

    if not submitted:
        return
    if not question.strip():
        st.warning("Please enter a question.")
        return

    try:
        with st.spinner("Searching the knowledge base and generating an answer..."):
            result = get_api_client().ask(question)
    except APIClientError as exc:
        st.error(str(exc))
        return

    render_response(result)


main()
