"""Chain-prompt constructions.

`build_chain_fixed` and `build_chain_degenerate` are copied VERBATIM from the
historical `same_window_deconfound.py` (research/Determinism/, seed-20260705
script) so the campaign reproduces the exact original constructions.
`build_chain_padded` is the new length-matched control: identical to `fixed`
except that the problem statement in steps >= 2 is replaced by task-irrelevant,
digit-free filler of matched character length.

All constructions share the identical step-1 prompt; at k == 1 every
construction collapses to the same single `raw_prompt`, so k = 1 cells measure
within-window serving noise only.
"""
from __future__ import annotations

_FILLER_BASE = (
    "Keep your intermediate results organized and double-check each arithmetic "
    "operation before moving on. Precision at every step matters more than "
    "speed in long calculations. Write clearly so the next step can follow. "
)


def raw_prompt(question: str) -> str:
    """Single-step prompt (k=1 reference; shared by all constructions)."""
    return (
        "Solve this math problem step by step. End your response with '#### ' "
        f"followed by the final numerical answer.\n\nProblem: {question}"
    )


def filler_text(length: int) -> str:
    """Deterministic, digit-free filler of exactly `length` characters."""
    reps = (length // len(_FILLER_BASE)) + 1
    return (_FILLER_BASE * reps)[:length]


def build_chain_fixed(question: str, k: int) -> list[str]:
    """Problem preserved in every step (verbatim from same_window_deconfound.py)."""
    chain = [
        f"Read this math problem and identify what quantities are given and "
        f"what is being asked. Do NOT solve it yet.\n\nProblem: {question}",
    ]
    for i in range(1, k - 1):
        chain.append(
            f"Original problem (do not lose sight of this):\n{question}\n\n"
            "Your prior reasoning steps produced:\n{PREV_OUTPUT}\n\n"
            f"Now perform step {i+1}: work through the next part of the "
            "calculation. Show your work but do NOT give the final answer yet."
        )
    chain.append(
        f"Original problem (the question you are answering):\n{question}\n\n"
        "Your prior reasoning steps produced:\n{PREV_OUTPUT}\n\n"
        "Now compute the final answer to the original problem. End with "
        "'#### ' followed by the numerical answer."
    )
    return chain


def build_chain_degenerate(question: str, k: int) -> list[str]:
    """The BUG: only step 1 gets the problem (verbatim from same_window_deconfound.py)."""
    chain = [
        f"Read this math problem and identify what quantities are given and "
        f"what is being asked. Do NOT solve it yet.\n\nProblem: {question}",
    ]
    for i in range(1, k - 1):
        chain.append(
            "Your prior reasoning steps produced:\n{PREV_OUTPUT}\n\n"
            f"Now perform step {i+1}: work through the next part of the "
            "calculation. Show your work but do NOT give the final answer yet."
        )
    chain.append(
        "Your prior reasoning steps produced:\n{PREV_OUTPUT}\n\n"
        "Now compute the final answer. End with '#### ' followed by the "
        "numerical answer."
    )
    return chain


def build_chain_padded(question: str, k: int) -> list[str]:
    """Length control: `fixed` with the problem replaced by matched-length filler."""
    filler = filler_text(len(question))
    chain = [
        f"Read this math problem and identify what quantities are given and "
        f"what is being asked. Do NOT solve it yet.\n\nProblem: {question}",
    ]
    for i in range(1, k - 1):
        chain.append(
            f"Note (general guidance, not the task):\n{filler}\n\n"
            "Your prior reasoning steps produced:\n{PREV_OUTPUT}\n\n"
            f"Now perform step {i+1}: work through the next part of the "
            "calculation. Show your work but do NOT give the final answer yet."
        )
    chain.append(
        f"Note (general guidance, not the task):\n{filler}\n\n"
        "Your prior reasoning steps produced:\n{PREV_OUTPUT}\n\n"
        "Now compute the final answer to the original problem. End with "
        "'#### ' followed by the numerical answer."
    )
    return chain


def filler_words(n_words: int) -> str:
    """Digit-free filler with exactly `n_words` whitespace tokens."""
    base = _FILLER_BASE.split()
    out: list[str] = []
    while len(out) < n_words:
        out.extend(base)
    return " ".join(out[:max(1, n_words)])


_PAD2_HEADER = "Note (general guidance, not the task):"


def _pad2_block(block: str) -> str:
    """Replace a problem-bearing block with a token-count-matched note."""
    n = len(block.split())
    header_n = len(_PAD2_HEADER.split())
    return _PAD2_HEADER + "\n" + filler_words(max(1, n - header_n))


def build_chain_padded2(question: str, k: int) -> list[str]:
    """R2 length control: `fixed` with each problem block replaced by filler
    matched in whitespace-token count (see EXPERIMENT_PLAN.md Addendum R2)."""
    fixed = build_chain_fixed(question, k)
    mid_block = f"Original problem (do not lose sight of this):\n{question}"
    fin_block = f"Original problem (the question you are answering):\n{question}"
    out = [fixed[0]]
    for tmpl in fixed[1:-1]:
        out.append(tmpl.replace(mid_block, _pad2_block(mid_block)))
    out.append(fixed[-1].replace(fin_block, _pad2_block(fin_block)))
    return out


def build_chain_finalonly(question: str, k: int) -> list[str]:
    """R2 fresh-solve probe: degenerate chain whose FINAL step re-anchors the
    problem (identical to `fixed`'s final step)."""
    chain = build_chain_degenerate(question, k)
    chain[-1] = build_chain_fixed(question, k)[-1]
    return chain


BUILDERS = {
    "fixed": build_chain_fixed,
    "degenerate": build_chain_degenerate,
    "padded": build_chain_padded,
    "padded2": build_chain_padded2,
    "finalonly": build_chain_finalonly,
}


def render_step(template: str, prev_output: str) -> str:
    return template.replace("{PREV_OUTPUT}", prev_output)
