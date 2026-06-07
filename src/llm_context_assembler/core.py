"""Assemble LLM context from prioritized sources within a token budget.

:class:`ContextAssembler` holds named :class:`ContextSource` objects and
picks which ones fit inside a token budget.  Sources are included in
descending priority order; lower-priority sources are dropped when the
budget is exceeded.

A *tokenizer* is any callable that maps a string to an integer token count.
The default is ``lambda s: len(s) // 4`` (a rough chars/4 approximation).

Example::

    assembler = ContextAssembler(budget=2000)
    assembler.add_source("system",   "You are a helpful assistant.", priority=100)
    assembler.add_source("docs",     long_retrieved_text,            priority=50)
    assembler.add_source("history",  recent_conversation,            priority=60)

    result = assembler.assemble()
    for src in result.included:
        print(src.name, src.estimated_tokens)

    combined = result.content()   # all included sources joined
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


# Default tokenizer: rough chars / 4 approximation.
def _default_tokenizer(text: str) -> int:
    return max(1, len(text) // 4)


class SourceNotFoundError(KeyError):
    """Raised when a source name is not registered."""

    def __init__(self, name: str) -> None:
        self.name = name
        super().__init__(f"Source {name!r} not found.")


class DuplicateSourceError(ValueError):
    """Raised when a source name is already registered."""

    def __init__(self, name: str) -> None:
        self.name = name
        super().__init__(f"Source {name!r} is already registered.")


@dataclass
class ContextSource:
    """A single named context source.

    Attributes:
        name:             Unique identifier.
        content:          The text content of this source.
        priority:         Higher = included first when budget is tight.
        estimated_tokens: Token count as estimated by the assembler's
                          tokenizer at insertion time.
        metadata:         Arbitrary extra data.
    """

    name: str
    content: str
    priority: int = 0
    estimated_tokens: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict."""
        return {
            "name": self.name,
            "content": self.content,
            "priority": self.priority,
            "estimated_tokens": self.estimated_tokens,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ContextSource:
        """Reconstruct a :class:`ContextSource` from a plain dict."""
        return cls(
            name=data["name"],
            content=data["content"],
            priority=int(data.get("priority", 0)),
            estimated_tokens=int(data.get("estimated_tokens", 0)),
            metadata=dict(data.get("metadata", {})),
        )

    def __repr__(self) -> str:
        return (
            f"ContextSource(name={self.name!r},"
            f" priority={self.priority},"
            f" tokens={self.estimated_tokens})"
        )


@dataclass
class AssemblyResult:
    """Result of a :meth:`~ContextAssembler.assemble` call.

    Attributes:
        included:    Sources that fit within the budget, in the order
                     they were selected (descending priority, FIFO tiebreak).
        excluded:    Sources that did not fit (overflow).
        total_tokens: Total estimated tokens of *included* sources.
        budget:      The budget that was used.
    """

    included: list[ContextSource]
    excluded: list[ContextSource]
    total_tokens: int
    budget: int

    def content(self, *, separator: str = "\n\n") -> str:
        """Concatenate all included sources with *separator*."""
        return separator.join(s.content for s in self.included)

    def fits_all(self) -> bool:
        """Return ``True`` if every source was included."""
        return len(self.excluded) == 0

    def budget_remaining(self) -> int:
        """Tokens remaining after including all included sources."""
        return max(0, self.budget - self.total_tokens)

    def __repr__(self) -> str:
        return (
            f"AssemblyResult(included={len(self.included)},"
            f" excluded={len(self.excluded)},"
            f" tokens={self.total_tokens}/{self.budget})"
        )


class ContextAssembler:
    """Assemble LLM context from prioritized named sources.

    Sources are included in descending priority order until the *budget*
    (in tokens) is exhausted.  Ties in priority are broken by insertion
    order (FIFO).

    Args:
        budget:    Maximum total token count for all included sources.
        tokenizer: Callable mapping a string to its integer token count.
                   Defaults to ``len(text) // 4`` (chars-per-4 estimate).

    Example::

        asm = ContextAssembler(budget=1000)
        asm.add_source("system", "Be helpful.", priority=100)
        asm.add_source("doc",    big_document,  priority=10)

        result = asm.assemble()
        prompt = result.content()
    """

    def __init__(
        self,
        *,
        budget: int,
        tokenizer: Callable[[str], int] | None = None,
    ) -> None:
        self._budget = budget
        self._tokenizer = tokenizer or _default_tokenizer
        self._sources: dict[str, ContextSource] = {}
        self._order: list[str] = []

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_source(
        self,
        name: str,
        content: str,
        *,
        priority: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> ContextSource:
        """Register a context source.

        Args:
            name:     Unique identifier.
            content:  Text content.
            priority: Higher = included first when budget is tight.
            metadata: Arbitrary extra data.

        Returns:
            The new :class:`ContextSource`.

        Raises:
            DuplicateSourceError: If *name* is already registered.
        """
        if name in self._sources:
            raise DuplicateSourceError(name)
        tokens = self._tokenizer(content)
        src = ContextSource(
            name=name,
            content=content,
            priority=priority,
            estimated_tokens=tokens,
            metadata=dict(metadata or {}),
        )
        self._sources[name] = src
        self._order.append(name)
        return src

    def update_source(
        self,
        name: str,
        content: str,
        *,
        priority: int | None = None,
    ) -> ContextSource:
        """Replace the content of an existing source.

        Also recalculates *estimated_tokens*.  Optionally updates *priority*.

        Raises:
            SourceNotFoundError: If *name* is not registered.
        """
        src = self.get(name)
        src.content = content
        src.estimated_tokens = self._tokenizer(content)
        if priority is not None:
            src.priority = priority
        return src

    def remove_source(self, name: str) -> None:
        """Remove a source.

        Raises:
            SourceNotFoundError: If *name* is not registered.
        """
        if name not in self._sources:
            raise SourceNotFoundError(name)
        del self._sources[name]
        self._order.remove(name)

    # ------------------------------------------------------------------
    # Assembly
    # ------------------------------------------------------------------

    def assemble(self) -> AssemblyResult:
        """Pick sources that fit within the budget.

        Returns an :class:`AssemblyResult` with *included* in the order
        sources were selected (descending priority, FIFO tiebreak).

        The order in the result reflects the *selection* order, not
        necessarily the semantic order.  Callers may re-sort *included*
        if a different presentation order is needed.
        """
        # Sort by (priority desc, insertion index asc) — stable
        ordered = sorted(
            enumerate(self._order),
            key=lambda x: (-self._sources[x[1]].priority, x[0]),
        )
        included: list[ContextSource] = []
        excluded: list[ContextSource] = []
        tokens_used = 0
        for _, name in ordered:
            src = self._sources[name]
            if tokens_used + src.estimated_tokens <= self._budget:
                included.append(src)
                tokens_used += src.estimated_tokens
            else:
                excluded.append(src)
        return AssemblyResult(
            included=included,
            excluded=excluded,
            total_tokens=tokens_used,
            budget=self._budget,
        )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get(self, name: str) -> ContextSource:
        """Return source by *name*.

        Raises:
            SourceNotFoundError: If not found.
        """
        if name not in self._sources:
            raise SourceNotFoundError(name)
        return self._sources[name]

    def all(self) -> list[ContextSource]:
        """All sources in insertion order."""
        return [self._sources[n] for n in self._order]

    def total_tokens(self) -> int:
        """Sum of *estimated_tokens* for all registered sources."""
        return sum(s.estimated_tokens for s in self._sources.values())

    def fits_all(self) -> bool:
        """Return ``True`` if all sources fit within the budget."""
        return self.total_tokens() <= self._budget

    def budget_remaining(self) -> int:
        """Budget minus total tokens of all sources (may be negative)."""
        return self._budget - self.total_tokens()

    @property
    def budget(self) -> int:
        """The configured token budget."""
        return self._budget

    def set_budget(self, budget: int) -> None:
        """Update the token budget."""
        self._budget = budget

    def __len__(self) -> int:
        return len(self._sources)

    def __contains__(self, name: str) -> bool:
        return name in self._sources

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Remove all sources."""
        self._sources.clear()
        self._order.clear()

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict."""
        return {
            "budget": self._budget,
            "order": list(self._order),
            "sources": [s.to_dict() for s in self._sources.values()],
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        tokenizer: Callable[[str], int] | None = None,
    ) -> ContextAssembler:
        """Reconstruct a :class:`ContextAssembler` from a plain dict."""
        asm = cls(budget=int(data["budget"]), tokenizer=tokenizer)
        for d in data.get("sources", []):
            src = ContextSource.from_dict(d)
            asm._sources[src.name] = src
        # Rebuild _order so it always matches _sources exactly, regardless of
        # any inconsistency in the serialised data.  Honour the recorded order
        # for known names, drop dangling names, then append any sources that
        # were missing from "order" (in their serialised order).
        seen: set[str] = set()
        order: list[str] = []
        for name in data.get("order", []):
            if name in asm._sources and name not in seen:
                order.append(name)
                seen.add(name)
        for name in asm._sources:
            if name not in seen:
                order.append(name)
                seen.add(name)
        asm._order = order
        return asm

    def __repr__(self) -> str:
        return (
            f"ContextAssembler(sources={len(self._sources)},"
            f" budget={self._budget},"
            f" total_tokens={self.total_tokens()})"
        )
