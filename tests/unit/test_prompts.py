"""Grounding prompt behavior."""

import json

from app.rag.prompts import GROUNDING_INSTRUCTIONS, build_grounded_input
from tests.helpers import make_retrieved_chunk


def test_prompt_labels_every_context_source() -> None:
    rendered = build_grounded_input(
        "What happened?",
        [
            make_retrieved_chunk(chunk_id="a", text="First"),
            make_retrieved_chunk(chunk_id="b", text="Second"),
        ],
    )
    context = json.loads(rendered.split("Retrieved context (JSON reference data):\n", 1)[1])
    assert [item["source_id"] for item in context] == ["S1", "S2"]
    assert context[1]["content"] == "Second"


def test_instructions_require_grounding_and_treat_context_as_untrusted() -> None:
    lowered = GROUNDING_INSTRUCTIONS.lower()
    assert "only" in lowered
    assert "insufficient" in lowered
    assert "source" in lowered
    assert "untrusted" in lowered

