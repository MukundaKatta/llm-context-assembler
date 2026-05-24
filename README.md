# llm-context-assembler

Assemble LLM context from prioritized sources within a token budget.

Add named sources (system prompt, retrieved docs, conversation history, etc.) with priorities. Call `assemble()` to get the sources that fit, dropping lower-priority ones when the budget is exceeded.

## Install

```bash
pip install llm-context-assembler
```

## Quick start

```python
from llm_context_assembler import ContextAssembler

asm = ContextAssembler(budget=2000)
asm.add_source("system",   "You are a helpful assistant.",  priority=100)
asm.add_source("history",  conversation_history,            priority=60)
asm.add_source("docs",     retrieved_documents,             priority=40)
asm.add_source("context",  extra_context,                   priority=20)

result = asm.assemble()
print(f"Included {len(result.included)} of {len(asm)} sources")
print(f"Tokens used: {result.total_tokens}/{result.budget}")

# Combine for the LLM call
combined_context = result.content(separator="\n\n")
```

## API

### `ContextAssembler`

```python
ContextAssembler(*, budget: int, tokenizer: Callable[[str], int] | None = None)
```

Default tokenizer: `len(text) // 4` (chars-per-4 approximation). Pass your own for exact counts.

| Method | Description |
|---|---|
| `add_source(name, content, *, priority, metadata)` | Register a source |
| `update_source(name, content, *, priority)` | Replace source content |
| `remove_source(name)` | Remove a source |
| `assemble()` | Pick sources that fit the budget → `AssemblyResult` |
| `get(name)` | Get source by name |
| `all()` | All sources in insertion order |
| `total_tokens()` | Sum of all source token estimates |
| `fits_all()` | Do all sources fit in budget? |
| `budget_remaining()` | Budget minus total tokens |
| `set_budget(n)` | Update the budget |
| `clear()` | Remove all sources |
| `to_dict()` / `from_dict(data)` | Serialise/restore |

### `AssemblyResult`

| Property/Method | Description |
|---|---|
| `included` | Sources that fit (priority order) |
| `excluded` | Sources that didn't fit |
| `total_tokens` | Tokens used by included sources |
| `budget` | Budget used for this assembly |
| `content(*, separator)` | Join included source contents |
| `fits_all()` | Were all sources included? |
| `budget_remaining()` | Tokens left after included sources |

## License

MIT
