"""Tokenizer for mixed Chinese/English text: English words + Chinese unigrams/bigrams."""

from __future__ import annotations

import re

_EN = re.compile(r"[a-zA-Z0-9]+")
_ZH = re.compile(r"[\u4e00-\u9fff]+")


def tokenize(text: str, zh_bigrams: bool = True) -> list[str]:
    """Lowercased English words + Chinese characters (plus bigrams by default).

    Bigrams give BM25 a fighting chance against words-segmentation-free Chinese,
    since 「检索」 as a bigram is far more discriminative than two unigrams.
    """
    tokens: list[str] = []
    for m in _EN.finditer(text):
        tokens.append(m.group(0).lower())
    for m in _ZH.finditer(text):
        chars = list(m.group(0))
        tokens.extend(chars)
        if zh_bigrams:
            tokens.extend(a + b for a, b in zip(chars, chars[1:]))
    return tokens
