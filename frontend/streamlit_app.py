"""Polished Streamlit product demo for the OpsRAG FastAPI service."""

from html import escape

import streamlit as st

from app.api.schemas import AskResponse, SourceResponse
from frontend.client import APIClientError, OpsRAGAPIClient
from frontend.config import get_frontend_settings
from frontend.theme import APP_CSS

EXAMPLE_QUESTIONS = (
    "Why does PostgreSQL return connection refused in Docker Compose?",
    "How should I troubleshoot an HTTP 503 error?",
    "What should I check when a FastAPI application fails to start?",
    "How should environment variables and secrets be handled in production?",
)


@st.cache_resource
def get_api_client() -> OpsRAGAPIClient:
    """Reuse the HTTP connection pool across Streamlit reruns."""

    settings = get_frontend_settings()
    return OpsRAGAPIClient(
        base_url=settings.api_base_url,
        timeout_seconds=settings.api_timeout_seconds,
    )


def apply_example_question(question: str) -> None:
    """Copy an example into the question field before the next rerun."""

    st.session_state["question_input"] = question


def render_navigation() -> None:
    """Render a lightweight product-style navigation bar."""

    st.markdown(
        """
        <div class="ops-nav">
            <div class="ops-brand">
                <span class="ops-brand-mark">O</span>
                <span>OpsRAG</span>
            </div>
            <div class="ops-nav-meta">
                <span class="ops-live-dot"></span>
                Local-first technical assistant
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    """Introduce the product and its core trust signals."""

    st.markdown(
        """
        <section class="ops-hero">
            <div class="ops-eyebrow">Evidence-first operations support</div>
            <h1>Resolve incidents with<br><span class="ops-gradient-text">grounded answers.</span></h1>
            <p class="ops-hero-copy">
                Ask technical operations questions in natural language. OpsRAG combines
                hybrid retrieval, reranking, and strict source attribution to turn a local
                knowledge base into answers you can inspect and trust.
            </p>
            <div class="ops-proof-row">
                <span class="ops-proof"><span class="ops-check">✓</span> Hybrid retrieval</span>
                <span class="ops-proof"><span class="ops-check">✓</span> Cross-encoder reranking</span>
                <span class="ops-proof"><span class="ops-check">✓</span> Validated citations</span>
                <span class="ops-proof"><span class="ops-check">✓</span> Provider-neutral LLM</span>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_pipeline_card() -> None:
    """Explain the bounded answer flow beside the question form."""

    st.markdown(
        """
        <div class="ops-flow-card">
            <div class="ops-flow-title">How your answer is built</div>
            <div class="ops-flow-step">
                <div class="ops-flow-number">01</div>
                <div><strong>Understand</strong><span>Route the question to direct or knowledge-grounded generation.</span></div>
            </div>
            <div class="ops-flow-step">
                <div class="ops-flow-number">02</div>
                <div><strong>Retrieve</strong><span>Combine semantic and exact-keyword evidence from the knowledge base.</span></div>
            </div>
            <div class="ops-flow-step">
                <div class="ops-flow-number">03</div>
                <div><strong>Rerank</strong><span>Promote the most relevant contexts before generation.</span></div>
            </div>
            <div class="ops-flow-step">
                <div class="ops-flow-number">04</div>
                <div><strong>Validate</strong><span>Return only source identifiers that were actually supplied to the model.</span></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_question_workspace() -> tuple[bool, str]:
    """Render the question form and selectable example prompts."""

    st.markdown('<div class="ops-section-kicker">Knowledge workspace</div>', unsafe_allow_html=True)
    st.markdown('<div class="ops-section-title">Ask OpsRAG</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="ops-section-copy">Use English or Turkish. Detailed questions usually produce stronger evidence.</p>',
        unsafe_allow_html=True,
    )

    with st.form("ask_form"):
        question = st.text_area(
            "Technical question",
            key="question_input",
            max_chars=4000,
            placeholder="Describe the issue, error code, or behavior you want to investigate...",
        )
        submitted = st.form_submit_button(
            "Generate grounded answer  →",
            key="ask_button",
            type="primary",
            width="stretch",
        )

    st.markdown('<div class="ops-example-label">Try an example</div>', unsafe_allow_html=True)
    for row_start in range(0, len(EXAMPLE_QUESTIONS), 2):
        columns = st.columns(2, gap="small")
        for offset, column in enumerate(columns):
            example_index = row_start + offset
            if example_index >= len(EXAMPLE_QUESTIONS):
                continue
            example = EXAMPLE_QUESTIONS[example_index]
            column.button(
                example,
                key=f"example_{example_index}",
                on_click=apply_example_question,
                args=(example,),
                width="stretch",
            )

    return submitted, question


def render_source(source: SourceResponse) -> None:
    """Render one API-owned source record without trusting model metadata."""

    with st.expander(f"[{source.source_id}] {source.title}"):
        document, location = st.columns([1.2, 1])
        document.markdown(f"**Document**  \n`{source.document}`")
        location_text = source.section
        if source.page_number is not None:
            location_text = f"{location_text} · page {source.page_number}"
        location.markdown(f"**Location**  \n{location_text}")
        st.markdown(f"**Relevance · {source.score:.1%}**")
        st.progress(source.score)
        st.caption(f"Contributing chunks: {', '.join(source.chunk_ids)}")


def render_response(result: AskResponse) -> None:
    """Render the answer, confidence, execution metadata, and sources."""

    st.divider()
    st.markdown('<div class="ops-section-kicker">Generated result</div>', unsafe_allow_html=True)
    st.markdown('<div class="ops-section-title">Evidence-backed answer</div>', unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown(result.answer)
        st.write("")
        confidence, route, contexts = st.columns(3)
        confidence.metric("Retrieval confidence", f"{result.retrieval_confidence:.1%}")
        route.metric("Route", result.metadata.route.title())
        contexts.metric("Retrieved contexts", result.metadata.retrieved_chunks)
        if result.metadata.route == "knowledge":
            st.progress(result.retrieval_confidence)
            st.caption(
                "Confidence is a retrieval relevance heuristic, not a calibrated probability."
            )

    st.markdown('<div class="ops-section-kicker" style="margin-top:1.7rem">Evidence</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="ops-section-title">Sources · {len(result.sources)}</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        f"Retrieval method: {result.metadata.retrieval_method.replace('_', ' ')} · "
        f"Cited sources: {result.metadata.cited_sources}"
    )
    if not result.sources:
        st.info("No knowledge-base sources were used for this response.")
        return
    for source in result.sources:
        render_source(source)


def render_footer() -> None:
    """Close the page with a compact product footer."""

    settings = get_frontend_settings()
    safe_api_base_url = escape(settings.api_base_url)
    st.markdown(
        f"""
        <footer class="ops-footer">
            <span>OpsRAG · Source-grounded technical intelligence</span>
            <span>API · {safe_api_base_url}</span>
        </footer>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    """Render the product demo and execute questions on form submission."""

    st.set_page_config(
        page_title="OpsRAG · Grounded Technical Answers",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(APP_CSS, unsafe_allow_html=True)
    render_navigation()
    render_hero()

    workspace, explainer = st.columns([1.55, 0.85], gap="large", vertical_alignment="top")
    with workspace:
        submitted, question = render_question_workspace()
    with explainer:
        render_pipeline_card()

    if submitted:
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            try:
                with st.spinner("Retrieving evidence and generating a grounded answer..."):
                    result = get_api_client().ask(question)
            except APIClientError as exc:
                st.error(str(exc))
            else:
                render_response(result)

    render_footer()


main()
