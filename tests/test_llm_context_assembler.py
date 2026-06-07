"""Tests for llm-context-assembler."""

from __future__ import annotations

import pytest

from llm_context_assembler import ContextAssembler, ContextSource
from llm_context_assembler.core import DuplicateSourceError, SourceNotFoundError


def _exact_tokens(text: str) -> int:
    """Exact char-count tokenizer for deterministic tests."""
    return len(text)


# ---------------------------------------------------------------------------
# ContextSource — construction and serialisation
# ---------------------------------------------------------------------------


def test_context_source_minimal():
    src = ContextSource(name="s1", content="Hello.")
    assert src.name == "s1"
    assert src.content == "Hello."
    assert src.priority == 0
    assert src.metadata == {}


def test_context_source_to_dict():
    src = ContextSource(
        name="doc",
        content="Body text.",
        priority=5,
        estimated_tokens=10,
        metadata={"url": "http://x"},
    )
    d = src.to_dict()
    assert d["name"] == "doc"
    assert d["content"] == "Body text."
    assert d["priority"] == 5
    assert d["estimated_tokens"] == 10
    assert d["metadata"] == {"url": "http://x"}


def test_context_source_from_dict_round_trip():
    src = ContextSource(
        name="s",
        content="X",
        priority=3,
        estimated_tokens=1,
        metadata={"k": "v"},
    )
    restored = ContextSource.from_dict(src.to_dict())
    assert restored.name == src.name
    assert restored.content == src.content
    assert restored.priority == src.priority
    assert restored.estimated_tokens == src.estimated_tokens
    assert restored.metadata == src.metadata


def test_context_source_repr():
    src = ContextSource(name="docs", content="Hi.", priority=5, estimated_tokens=3)
    r = repr(src)
    assert "docs" in r
    assert "5" in r


# ---------------------------------------------------------------------------
# ContextAssembler — add_source
# ---------------------------------------------------------------------------


def test_add_source():
    asm = ContextAssembler(budget=1000, tokenizer=_exact_tokens)
    src = asm.add_source("s1", "Hello.")
    assert src.name == "s1"
    assert "s1" in asm


def test_add_source_estimates_tokens():
    asm = ContextAssembler(budget=1000, tokenizer=_exact_tokens)
    src = asm.add_source("s1", "Hello.")
    assert src.estimated_tokens == len("Hello.")


def test_add_source_with_priority():
    asm = ContextAssembler(budget=1000, tokenizer=_exact_tokens)
    src = asm.add_source("s1", "Hi.", priority=10)
    assert src.priority == 10


def test_add_source_with_metadata():
    asm = ContextAssembler(budget=1000, tokenizer=_exact_tokens)
    src = asm.add_source("s1", "Hi.", metadata={"url": "http://x"})
    assert src.metadata == {"url": "http://x"}


def test_add_duplicate_raises():
    asm = ContextAssembler(budget=1000, tokenizer=_exact_tokens)
    asm.add_source("s1", "A")
    with pytest.raises(DuplicateSourceError) as exc_info:
        asm.add_source("s1", "B")
    assert exc_info.value.name == "s1"


# ---------------------------------------------------------------------------
# ContextAssembler — update_source / remove_source
# ---------------------------------------------------------------------------


def test_update_source():
    asm = ContextAssembler(budget=1000, tokenizer=_exact_tokens)
    asm.add_source("s1", "Short.")
    src = asm.update_source("s1", "Much longer content now.")
    assert src.content == "Much longer content now."
    assert src.estimated_tokens == len("Much longer content now.")


def test_update_source_priority():
    asm = ContextAssembler(budget=1000, tokenizer=_exact_tokens)
    asm.add_source("s1", "X.", priority=1)
    asm.update_source("s1", "X.", priority=99)
    assert asm.get("s1").priority == 99


def test_update_missing_raises():
    asm = ContextAssembler(budget=1000, tokenizer=_exact_tokens)
    with pytest.raises(SourceNotFoundError):
        asm.update_source("nope", "X.")


def test_remove_source():
    asm = ContextAssembler(budget=1000, tokenizer=_exact_tokens)
    asm.add_source("s1", "Hello.")
    asm.remove_source("s1")
    assert "s1" not in asm


def test_remove_missing_raises():
    asm = ContextAssembler(budget=1000, tokenizer=_exact_tokens)
    with pytest.raises(SourceNotFoundError):
        asm.remove_source("nope")


# ---------------------------------------------------------------------------
# ContextAssembler — assemble
# ---------------------------------------------------------------------------


def test_assemble_all_fit():
    asm = ContextAssembler(budget=1000, tokenizer=_exact_tokens)
    asm.add_source("s1", "Hello.")  # 6 tokens
    asm.add_source("s2", "World.")  # 6 tokens
    result = asm.assemble()
    assert result.fits_all()
    assert len(result.included) == 2
    assert result.excluded == []


def test_assemble_priority_order():
    asm = ContextAssembler(budget=1000, tokenizer=_exact_tokens)
    asm.add_source("low", "Low.", priority=1)
    asm.add_source("high", "High.", priority=10)
    result = asm.assemble()
    # Included in priority order: high first
    assert result.included[0].name == "high"
    assert result.included[1].name == "low"


def test_assemble_drops_overflow():
    # budget = 10 chars; each source is 6 chars → only one fits
    asm = ContextAssembler(budget=10, tokenizer=_exact_tokens)
    asm.add_source("s1", "Hello.", priority=10)  # 6 tokens
    asm.add_source("s2", "World.", priority=5)  # 6 tokens
    result = asm.assemble()
    assert len(result.included) == 1
    assert result.included[0].name == "s1"
    assert len(result.excluded) == 1
    assert result.excluded[0].name == "s2"


def test_assemble_fifo_tiebreak():
    asm = ContextAssembler(budget=1000, tokenizer=_exact_tokens)
    asm.add_source("first", "A.", priority=5)
    asm.add_source("second", "B.", priority=5)
    result = asm.assemble()
    assert result.included[0].name == "first"
    assert result.included[1].name == "second"


def test_assemble_empty():
    asm = ContextAssembler(budget=100, tokenizer=_exact_tokens)
    result = asm.assemble()
    assert result.included == []
    assert result.excluded == []
    assert result.total_tokens == 0


def test_assemble_exact_budget():
    # Exactly fills the budget
    asm = ContextAssembler(budget=6, tokenizer=_exact_tokens)
    asm.add_source("s1", "Hello.")  # exactly 6 tokens
    result = asm.assemble()
    assert result.fits_all()
    assert result.total_tokens == 6


def test_assemble_one_over_budget():
    asm = ContextAssembler(budget=5, tokenizer=_exact_tokens)
    asm.add_source("s1", "Hello.")  # 6 tokens > budget 5
    result = asm.assemble()
    assert len(result.included) == 0
    assert len(result.excluded) == 1


# ---------------------------------------------------------------------------
# AssemblyResult
# ---------------------------------------------------------------------------


def test_result_content_joined():
    asm = ContextAssembler(budget=1000, tokenizer=_exact_tokens)
    asm.add_source("s1", "Part one.", priority=10)
    asm.add_source("s2", "Part two.", priority=5)
    result = asm.assemble()
    combined = result.content()
    assert "Part one." in combined
    assert "Part two." in combined


def test_result_content_custom_separator():
    asm = ContextAssembler(budget=1000, tokenizer=_exact_tokens)
    asm.add_source("s1", "A", priority=10)
    asm.add_source("s2", "B", priority=5)
    result = asm.assemble()
    assert result.content(separator="---") == "A---B"


def test_result_budget_remaining():
    asm = ContextAssembler(budget=100, tokenizer=_exact_tokens)
    asm.add_source("s1", "Hi.")  # 3 tokens
    result = asm.assemble()
    assert result.budget_remaining() == 97


def test_result_repr():
    asm = ContextAssembler(budget=100, tokenizer=_exact_tokens)
    asm.add_source("s1", "Hi.")
    result = asm.assemble()
    r = repr(result)
    assert "AssemblyResult" in r


# ---------------------------------------------------------------------------
# ContextAssembler — queries
# ---------------------------------------------------------------------------


def test_total_tokens():
    asm = ContextAssembler(budget=1000, tokenizer=_exact_tokens)
    asm.add_source("s1", "Hello.")  # 6
    asm.add_source("s2", "Hi.")  # 3
    assert asm.total_tokens() == 9


def test_fits_all_true():
    asm = ContextAssembler(budget=1000, tokenizer=_exact_tokens)
    asm.add_source("s1", "Hi.")
    assert asm.fits_all()


def test_fits_all_false():
    asm = ContextAssembler(budget=2, tokenizer=_exact_tokens)
    asm.add_source("s1", "Hello.")  # 6 > 2
    assert not asm.fits_all()


def test_budget_remaining_positive():
    asm = ContextAssembler(budget=100, tokenizer=_exact_tokens)
    asm.add_source("s1", "Hi.")  # 3 tokens
    assert asm.budget_remaining() == 97


def test_budget_remaining_negative():
    asm = ContextAssembler(budget=2, tokenizer=_exact_tokens)
    asm.add_source("s1", "Hello.")  # 6 > 2
    assert asm.budget_remaining() < 0


def test_all_in_insertion_order():
    asm = ContextAssembler(budget=1000, tokenizer=_exact_tokens)
    asm.add_source("a", "A.")
    asm.add_source("b", "B.")
    names = [s.name for s in asm.all()]
    assert names == ["a", "b"]


def test_len():
    asm = ContextAssembler(budget=1000, tokenizer=_exact_tokens)
    asm.add_source("a", "A.")
    asm.add_source("b", "B.")
    assert len(asm) == 2


def test_set_budget():
    asm = ContextAssembler(budget=100, tokenizer=_exact_tokens)
    asm.set_budget(500)
    assert asm.budget == 500


# ---------------------------------------------------------------------------
# Serialisation round-trip
# ---------------------------------------------------------------------------


def test_to_dict_round_trip():
    asm = ContextAssembler(budget=500, tokenizer=_exact_tokens)
    asm.add_source("sys", "System prompt.", priority=100)
    asm.add_source("docs", "Retrieved doc.", priority=50)

    restored = ContextAssembler.from_dict(asm.to_dict(), tokenizer=_exact_tokens)
    assert len(restored) == 2
    assert restored.budget == 500
    result = restored.assemble()
    assert result.fits_all()


def test_clear():
    asm = ContextAssembler(budget=100, tokenizer=_exact_tokens)
    asm.add_source("s1", "Hi.")
    asm.clear()
    assert len(asm) == 0


# ---------------------------------------------------------------------------
# repr
# ---------------------------------------------------------------------------


def test_repr():
    asm = ContextAssembler(budget=100, tokenizer=_exact_tokens)
    asm.add_source("s1", "Hi.")
    r = repr(asm)
    assert "ContextAssembler" in r
    assert "100" in r


# ---------------------------------------------------------------------------
# Default tokenizer
# ---------------------------------------------------------------------------


def test_default_tokenizer_chars_per_four():
    # Default tokenizer is len(text) // 4.
    asm = ContextAssembler(budget=1000)
    src = asm.add_source("s1", "x" * 40)
    assert src.estimated_tokens == 10


def test_default_tokenizer_minimum_one():
    # Even empty/short content costs at least one token.
    asm = ContextAssembler(budget=1000)
    assert asm.add_source("empty", "").estimated_tokens == 1
    assert asm.add_source("short", "ab").estimated_tokens == 1


# ---------------------------------------------------------------------------
# from_dict robustness against inconsistent serialised data
# ---------------------------------------------------------------------------


def test_from_dict_drops_dangling_order_names():
    # An "order" entry with no matching source must not corrupt the assembler.
    data = {
        "budget": 100,
        "order": ["a", "ghost"],
        "sources": [{"name": "a", "content": "aa"}],
    }
    asm = ContextAssembler.from_dict(data, tokenizer=_exact_tokens)
    assert [s.name for s in asm.all()] == ["a"]
    # assemble() must not raise on the dangling name.
    assert [s.name for s in asm.assemble().included] == ["a"]


def test_from_dict_appends_sources_missing_from_order():
    # A source absent from "order" must still appear (in serialised order).
    data = {
        "budget": 100,
        "order": ["a"],
        "sources": [
            {"name": "a", "content": "aa"},
            {"name": "b", "content": "bb"},
        ],
    }
    asm = ContextAssembler.from_dict(data, tokenizer=_exact_tokens)
    assert len(asm) == 2
    assert [s.name for s in asm.all()] == ["a", "b"]
    assert {s.name for s in asm.assemble().included} == {"a", "b"}


def test_from_dict_order_matches_sources_invariant():
    # _order and _sources must always carry the same names.
    data = {
        "budget": 100,
        "order": ["b", "missing", "a"],
        "sources": [
            {"name": "a", "content": "aa"},
            {"name": "b", "content": "bb"},
            {"name": "c", "content": "cc"},
        ],
    }
    asm = ContextAssembler.from_dict(data, tokenizer=_exact_tokens)
    names = [s.name for s in asm.all()]
    # Known order honoured first (b, a), then leftover sources (c).
    assert names == ["b", "a", "c"]
    assert len(names) == len(asm)
