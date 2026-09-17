"""Shared BM25 tokenization settings for indexing and retrieval."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable

import bm25s


TOKEN_PATTERN = r"[0-9a-zA-ZÀ-ỹà-ỹ]+"
TOKEN_RE = re.compile(TOKEN_PATTERN, re.UNICODE)


def normalize_for_bm25(text: str) -> str:
    """Apply the normalization used by both corpus and query tokenization."""
    return unicodedata.normalize("NFC", text).lower()


def iter_tokens(text: str) -> Iterable[str]:
    return TOKEN_RE.findall(normalize_for_bm25(text))


def tokenize_queries(texts: list[str]):
    """Tokenize queries with exactly the settings used by the index builder."""
    return bm25s.tokenize(
        [normalize_for_bm25(text) for text in texts],
        lower=False,
        stopwords=None,
        token_pattern=TOKEN_PATTERN,
        show_progress=False,
    )
