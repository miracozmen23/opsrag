"""Grounding prompt behavior."""

import json

from app.rag.attribution import build_source_contexts
from app.rag.prompts import GROUNDING_INSTRUCTIONS, build_grounded_input
from tests.helpers import make_retrieved_chunk


def test_prompt_labels_every_context_source() -> None:
    contexts = build_source_contexts(
        [
            make_retrieved_chunk(chunk_id="a", text="First"),
            make_retrieved_chunk(
                chunk_id="b",
                source="second.md",
                text="Second",
            ),
        ]
    )
    rendered = build_grounded_input("What happened?", contexts)
    context = json.loads(rendered.split("Retrieved context (JSON reference data):\n", 1)[1])
    assert [item["source_id"] for item in context] == ["S1", "S2"]
    assert context[1]["document"] == "second.md"
    assert context[1]["excerpts"][0]["content"] == "Second"


def test_instructions_require_grounding_and_treat_context_as_untrusted() -> None:
    lowered = GROUNDING_INSTRUCTIONS.lower()
    assert "only" in lowered
    assert "insufficient" in lowered
    assert "source" in lowered
    assert "exact source identifiers" in lowered
    assert "at least one" in lowered
    assert "untrusted" in lowered
