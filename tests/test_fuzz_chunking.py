"""Property-based fuzz: recursive chunking must exactly partition the source text.

Run with more trials: pytest tests/test_fuzz_chunking.py --trials 2000
"""

import random

import pytest

from rag_forge.chunking import fixed_chunks, recursive_chunks


def _run(trials: int) -> None:
    rng = random.Random(0)
    alphabet = ["句", "。", "；", "\n", "a", " ", "bb"]
    for trial in range(trials):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 400)))
        for size in (10, 30, 100):
            chunks = recursive_chunks("x", text, size=size, overlap=0)
            # exact partition: chunking must never lose or duplicate a character
            assert "".join(c.text for c in chunks) == text, (trial, size)
            # content fits the budget; boundary whitespace may ride along
            assert all(len(c.text.strip()) <= size for c in chunks), (trial, size)


def test_partition_invariant():
    _run(200)


@pytest.mark.parametrize("size,overlap", [(50, 10), (100, 0), (30, 29)])
def test_fixed_chunks_cover_document(size, overlap):
    rng = random.Random(1)
    text = "".join(rng.choice(["字", "。", "x"]) for _ in range(rng.randint(1, 600)))
    chunks = fixed_chunks("x", text, size=size, overlap=overlap)
    assert chunks[0].text == text[:size]
    assert text[-1] in chunks[-1].text  # document tail is covered
    starts = [c.start for c in chunks]
    assert starts == sorted(starts)  # monotonically advancing offsets
