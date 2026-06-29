"""
Module 4 — Positional Encoding.
===============================

By Module 3 we can turn tokens into vectors. But there's a problem waiting for
Module 5 (self-attention): attention looks at all tokens *at once*, as an
unordered **set**. It has no built-in idea of word ORDER.

Yet order changes meaning completely:
    "dog bites man"   ≠   "man bites dog"   (same words!)

**Positional encoding** fixes this by adding a small, position-dependent vector
to each token's embedding — so "dog" at position 0 looks different from "dog" at
position 2. The original Transformer uses fixed **sinusoidal** encodings:

    PE(pos, 2i)   = sin( pos / 10000^(2i/d) )
    PE(pos, 2i+1) = cos( pos / 10000^(2i/d) )

where `pos` = position in the sentence, `i` = dimension index, `d` = embedding
size. Low dimensions wiggle fast (fine position detail); high dimensions wiggle
slowly (coarse position) — together they give every position a unique fingerprint.

The final vector fed to the network is simply:

    input = token_embedding + positional_encoding

All plain functions; the token embedding reuses Module 1's helper (DRY).
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from app.core.text_pipeline import build_vocab, embedding, normalize_text, tokenize

MIN_DIM, MAX_DIM, DEFAULT_DIM = 4, 16, 8


# ---------------------------------------------------------------------------
# The core formula
# ---------------------------------------------------------------------------
def positional_encoding(num_positions: int, dim: int) -> np.ndarray:
    """Sinusoidal positional-encoding matrix of shape (num_positions, dim)."""
    pe = np.zeros((num_positions, dim))
    position = np.arange(num_positions)[:, None]              # (pos, 1)
    # Each dimension pair (2i, 2i+1) shares one frequency.
    div_term = np.power(10000.0, (2 * (np.arange(dim) // 2)) / dim)
    angles = position / div_term                              # (pos, dim)
    pe[:, 0::2] = np.sin(angles[:, 0::2])                     # even dims -> sin
    pe[:, 1::2] = np.cos(angles[:, 1::2])                     # odd dims  -> cos
    return pe


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na and nb else 0.0


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def run_positional(text: str, dim: int = DEFAULT_DIM) -> Dict[str, object]:
    """Compute positional encodings for the text and add them to embeddings."""
    dim = max(MIN_DIM, min(MAX_DIM, int(dim)))
    if dim % 2 != 0:
        dim += 1  # sinusoidal encoding needs an even number of dimensions

    tokens = tokenize(normalize_text(text)) or ["i", "love", "ai"]
    vocab = build_vocab(tokens)
    n = len(tokens)

    pe = positional_encoding(n, dim)

    per_token: List[Dict[str, object]] = []
    for pos, tok in enumerate(tokens):
        emb = np.array(embedding(vocab[tok], dim))
        pe_row = pe[pos]
        per_token.append(
            {
                "position": pos,
                "token": tok,
                "embedding": [round(float(x), 3) for x in emb],
                "positional": [round(float(x), 3) for x in pe_row],
                "result": [round(float(x), 3) for x in (emb + pe_row)],
            }
        )

    # How similar are two POSITION vectors to each other? (structure of order.)
    pos_similarity = [[round(cosine(pe[i], pe[j]), 3) for j in range(n)] for i in range(n)]

    return {
        "input": text,
        "dim": dim,
        "tokens": tokens,
        "pe_matrix": np.round(pe, 3).tolist(),
        "per_token": per_token,
        "position_similarity": pos_similarity,
        "worked_example": _worked_example(dim),
        "order_demo": _order_demo(dim),
        "explainers": _explainers(),
        "field_map": _field_map(),
        "code_sample": _code_sample(),
        "response": (
            f"I gave each of the {n} token(s) a unique {dim}-dimensional position "
            f"fingerprint and added it to the word embedding, so the model can now "
            f"tell where each word sits in the sentence."
        ),
    }


def _worked_example(dim: int) -> Dict[str, object]:
    """Spell out PE(pos=1, dim 0) and PE(pos=1, dim 1) from the formula."""
    pos = 1
    # dim 0 -> i=0 -> sin(pos / 10000^0) = sin(pos)
    # dim 1 -> i=0 -> cos(pos / 10000^0) = cos(pos)
    return {
        "pos": pos,
        "even": {
            "dim": 0,
            "formula": "PE(pos, 0) = sin(pos / 10000^(0/d)) = sin(pos)",
            "calc": f"sin({pos}) = {round(float(np.sin(pos)), 3)}",
        },
        "odd": {
            "dim": 1,
            "formula": "PE(pos, 1) = cos(pos / 10000^(0/d)) = cos(pos)",
            "calc": f"cos({pos}) = {round(float(np.cos(pos)), 3)}",
        },
        "note": (
            f"Higher dimensions divide pos by larger powers of 10000 (here d={dim}), "
            "so they change more slowly across positions."
        ),
    }


def _order_demo(dim: int) -> Dict[str, object]:
    """Show that swapping word order changes the position-aware vectors."""
    sentence_a = ["dog", "bites", "man"]
    sentence_b = ["man", "bites", "dog"]
    pe = positional_encoding(3, dim)

    def encode(words: List[str]) -> List[Dict[str, object]]:
        vocab = {w: i for i, w in enumerate(sorted(set(sentence_a + sentence_b)))}
        out = []
        for pos, w in enumerate(words):
            emb = np.array(embedding(vocab[w], dim))
            out.append({"token": w, "position": pos, "result": [round(float(x), 3) for x in (emb + pe[pos])]})
        return out

    return {
        "a": {"sentence": " ".join(sentence_a), "vectors": encode(sentence_a)},
        "b": {"sentence": " ".join(sentence_b), "vectors": encode(sentence_b)},
        "note": (
            "Same words, same embeddings — but because 'dog' and 'man' sit at "
            "different positions, their final vectors differ between the two "
            "sentences. That's how the model can tell who bit whom."
        ),
    }


# ---------------------------------------------------------------------------
# Teaching metadata
# ---------------------------------------------------------------------------
def _explainers() -> Dict[str, str]:
    return {
        "why": (
            "Why we need this: self-attention (next module) reads all tokens "
            "simultaneously as an unordered set, so on its own it can't tell "
            "'dog bites man' from 'man bites dog'. Positional encoding injects "
            "order back in."
        ),
        "how": (
            "How each value is computed: for a token at position `pos`, dimension "
            "`2i` uses sin(pos / 10000^(2i/d)) and dimension `2i+1` uses the "
            "matching cosine. Mixing many frequencies (fast-wiggling low dims, "
            "slow-wiggling high dims) gives every position a unique pattern, and "
            "the values stay neatly between -1 and 1."
        ),
        "added": (
            "How it's used: the positional vector is simply ADDED to the token "
            "embedding element-by-element. No extra parameters, no training — it's "
            "a fixed mathematical pattern the model learns to read."
        ),
    }


def _field_map() -> List[Dict[str, str]]:
    return [
        {
            "step": "Sinusoidal positional encoding",
            "field": "Deep Learning (Transformer architecture)",
            "why": "A design trick specific to Transformers — the model that powers modern LLMs.",
        },
        {
            "step": "Adding position to embeddings",
            "field": "Linear algebra / Deep Learning",
            "why": "Plain vector addition — preparing inputs for the attention layer.",
        },
    ]


def _code_sample() -> str:
    return (
        "# Sinusoidal positional encoding in PyTorch (the Transformer way):\n"
        "import torch, math\n"
        "\n"
        "def positional_encoding(seq_len, d_model):\n"
        "    pe = torch.zeros(seq_len, d_model)\n"
        "    pos = torch.arange(seq_len).unsqueeze(1).float()\n"
        "    div = torch.exp(torch.arange(0, d_model, 2).float()\n"
        "                    * (-math.log(10000.0) / d_model))\n"
        "    pe[:, 0::2] = torch.sin(pos * div)   # even dimensions\n"
        "    pe[:, 1::2] = torch.cos(pos * div)   # odd dimensions\n"
        "    return pe\n"
        "\n"
        "x = token_embeddings + positional_encoding(seq_len, d_model)\n"
        "\n"
        "# NOTE: many modern LLMs use LEARNED or rotary (RoPE) position encodings\n"
        "# instead, but the sinusoidal version is the classic, parameter-free one.\n"
    )
