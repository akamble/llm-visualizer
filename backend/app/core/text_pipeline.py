"""
Lesson 1 — How text becomes a response.
========================================

This module answers the very first question: *how is raw text converted into
numbers, pushed through some math, and turned back into a readable response?*

It is written in a **functional style**: each step is a small pure function that
takes input and returns output, with no hidden state. You can read them top to
bottom like a recipe:

    normalize -> tokenize -> build_vocab -> one_hot -> embedding -> hidden_layer

`run_pipeline()` at the bottom simply chains them together and packages every
intermediate value so the frontend can draw it.

Everything is deterministic: the same text always produces the same numbers,
because the "weights" and "embeddings" are seeded from the text itself instead
of being randomly initialised. (A real model *learns* these numbers from data;
here we fake them just so you can see the shapes and the math.)
"""

from __future__ import annotations

import re
from typing import Dict, List

import numpy as np

# --- Tunable, kept tiny on purpose so the vectors are readable on screen ------
EMBEDDING_DIM = 4   # each token becomes a dense vector of 4 numbers
HIDDEN_DIM = 3      # the hidden layer squashes those 4 numbers down to 3


# ---------------------------------------------------------------------------
# Step 1 — Normalize
# ---------------------------------------------------------------------------
def normalize_text(text: str) -> str:
    """Lowercase, collapse whitespace, and strip punctuation.

    Why: computers treat "AI", "ai" and "ai!" as three different things.
    Normalizing first means we don't waste vocabulary slots on those.
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)   # drop punctuation
    text = re.sub(r"\s+", " ", text).strip()    # collapse spaces
    return text


# ---------------------------------------------------------------------------
# Step 2 — Tokenize
# ---------------------------------------------------------------------------
def tokenize(text: str) -> List[str]:
    """Split a sentence into tokens (here, simply words).

    A *token* is the smallest unit the model works with. Real LLMs use
    'subword' tokens (e.g. "visualizing" -> "visual" + "izing"); we use whole
    words because they are easier to see.
    """
    if not text:
        return []
    return text.split(" ")


# ---------------------------------------------------------------------------
# Step 3 — Build the vocabulary (text -> numbers!)
# ---------------------------------------------------------------------------
def build_vocab(tokens: List[str]) -> Dict[str, int]:
    """Map each *unique* token to an integer ID.

    This is the crucial moment where **text turns into numbers**. The computer
    can't multiply the word "love", but it can multiply the number 2.
    IDs are assigned in order of first appearance.
    """
    vocab: Dict[str, int] = {}
    for tok in tokens:
        if tok not in vocab:
            vocab[tok] = len(vocab)
    return vocab


# ---------------------------------------------------------------------------
# Step 4 — One-hot encoding (a SPARSE vector)
# ---------------------------------------------------------------------------
def one_hot(token_id: int, vocab_size: int) -> List[int]:
    """Turn an ID into a vector that is all 0s except a single 1.

    Example with vocab_size = 4 and token_id = 2  ->  [0, 0, 1, 0]

    This is a *sparse* vector: mostly zeros. It's honest but wasteful — a
    50,000-word vocabulary means 50,000-long vectors. That's why we move to
    dense vectors next.
    """
    vec = [0] * vocab_size
    if 0 <= token_id < vocab_size:
        vec[token_id] = 1
    return vec


# ---------------------------------------------------------------------------
# Step 5 — Embedding (a DENSE vector)
# ---------------------------------------------------------------------------
def embedding(token_id: int, dim: int = EMBEDDING_DIM) -> List[float]:
    """Turn an ID into a short, dense vector of real numbers.

    A *dense* vector packs meaning into a few numbers (no wasted zeros).
    In a real model these numbers are *learned* so that similar words get
    similar vectors. Here we generate them deterministically from the ID so
    the lesson is reproducible.

    Think of it as the token's coordinates in a small "meaning space".
    """
    rng = np.random.default_rng(seed=1000 + token_id)
    vec = rng.standard_normal(dim)
    return [round(float(x), 3) for x in vec]


# ---------------------------------------------------------------------------
# Step 6 — Hidden layer:  h = activation(W · x + b)
# ---------------------------------------------------------------------------
def hidden_layer(vector: List[float]) -> Dict[str, object]:
    """Push a dense vector through one neural-network layer.

    The single most important formula in all of deep learning:

            h = activation(W · x + b)

    where
        x = input vector            (the embedding)
        W = weight matrix           (what the network "knows")
        b = bias vector             (a learnable offset)
        ·  = matrix multiplication  (every output mixes ALL inputs)
        activation = ReLU here      (keeps positives, zeroes out negatives)

    A *hidden layer* is "hidden" simply because it sits between the input and
    the output — you never see it directly, only its effect. Stack many of
    these and you get *deep* learning.
    """
    x = np.array(vector, dtype=float)
    in_dim = x.shape[0]

    # Deterministic, readable "weights" so the numbers are stable.
    rng = np.random.default_rng(seed=42)
    W = rng.standard_normal((HIDDEN_DIM, in_dim))   # shape: (out, in)
    b = rng.standard_normal(HIDDEN_DIM)

    z = W @ x + b                 # the linear part:  W · x + b
    h = np.maximum(0, z)          # ReLU activation: max(0, z)

    return {
        "weights": np.round(W, 3).tolist(),
        "bias": np.round(b, 3).tolist(),
        "pre_activation": np.round(z, 3).tolist(),   # z = W·x + b
        "output": np.round(h, 3).tolist(),           # h = ReLU(z)
        "formula": "h = ReLU(W · x + b)",
    }


# ---------------------------------------------------------------------------
# Step 7 — Co-occurrence grid (a Cartesian product)
# ---------------------------------------------------------------------------
def cooccurrence(tokens: List[str], vocab: Dict[str, int]) -> Dict[str, object]:
    """Build a vocab × vocab grid counting which tokens appear next to each other.

    The set of all (row, column) cells is the **Cartesian product** vocab × vocab:
    every word paired with every word. We then fill each cell with how often
    those two words were neighbours in the sentence.

    This is the seed idea behind how models learn that words relate to each other.
    """
    words = list(vocab.keys())
    size = len(words)
    matrix = np.zeros((size, size), dtype=int)

    for i in range(len(tokens) - 1):
        a = vocab[tokens[i]]
        b = vocab[tokens[i + 1]]
        matrix[a][b] += 1          # token a was followed by token b

    return {
        "labels": words,
        "matrix": matrix.tolist(),
        "total_pairs": int(matrix.sum()),
        "grid_size": f"{size} × {size} = {size * size} cells (Cartesian product)",
    }


# ---------------------------------------------------------------------------
# Orchestrator — chain every step together
# ---------------------------------------------------------------------------
def run_pipeline(text: str) -> Dict[str, object]:
    """Run the full text -> response pipeline and return every intermediate value."""
    normalized = normalize_text(text)
    tokens = tokenize(normalized)
    vocab = build_vocab(tokens)
    vocab_size = len(vocab)

    # Build the per-token table (the journey of each word).
    token_rows = []
    for tok in tokens:
        tid = vocab[tok]
        emb = embedding(tid)
        token_rows.append(
            {
                "token": tok,
                "id": tid,
                "one_hot": one_hot(tid, vocab_size),
                "embedding": emb,
                "hidden": hidden_layer(emb),
            }
        )

    grid = cooccurrence(tokens, vocab)

    # Step 8 — a human-readable "response" summarising what happened.
    response = (
        f'I read your text and found {len(tokens)} token(s) made of '
        f"{vocab_size} unique word(s). Each word was turned into a number, "
        f"then a {EMBEDDING_DIM}-dimensional dense vector, then passed through "
        f"a hidden layer to produce a {HIDDEN_DIM}-dimensional representation."
    )

    return {
        "input": text,
        "steps": {
            "1_normalized": normalized,
            "2_tokens": tokens,
            "3_vocab": vocab,
            "4_to_6_tokens": token_rows,   # one-hot, embedding, hidden per token
            "7_cooccurrence": grid,
        },
        "config": {"embedding_dim": EMBEDDING_DIM, "hidden_dim": HIDDEN_DIM},
        "explainers": _explainers(),
        "field_map": _field_map(),
        "response": response,
    }


# ---------------------------------------------------------------------------
# Teaching metadata — "how is this computed?" and "which field does it belong to?"
# ---------------------------------------------------------------------------
def _explainers() -> Dict[str, str]:
    """Plain-English explanations of HOW the dense/hidden numbers are produced."""
    return {
        "embedding": (
            "How each value is computed: we seed a random-number generator with "
            "the token's ID (seed = 1000 + id) and draw numbers from a 'standard "
            "normal' distribution (mean 0, spread 1), rounded to 3 decimals. "
            "Because the seed is fixed, the SAME word always gets the SAME vector. "
            "⚠️ This is a teaching stand-in: in a REAL model these numbers are NOT "
            "random — they are LEARNED from billions of words during training, so "
            "that words with similar meaning end up with similar vectors."
        ),
        "hidden": (
            "How each value is computed: h = ReLU(W · x + b). "
            "(1) Take one row of the weight matrix W and the embedding x, multiply "
            "them element-by-element and add up the results — that's a dot product. "
            "(2) Add the bias b for that row → this gives z (the 'pre-activation'). "
            "(3) Apply ReLU: keep z if it's positive, otherwise make it 0. "
            "Each output number therefore mixes ALL of the input numbers."
        ),
        "cooccurrence": (
            "How each cell is computed: we slide a window over the tokens and, for "
            "every ADJACENT pair, add 1 to cell [row word][next word]. We only count "
            "DIRECT neighbours. The last word in the sentence has nothing after it, "
            "so its entire row stays 0. A cell like 'ai → ai' is 0 simply because "
            "'ai' is never immediately followed by 'ai' in your text."
        ),
    }


def _field_map() -> List[Dict[str, str]]:
    """Which area of AI each step belongs to — to connect lessons to the big picture."""
    return [
        {
            "step": "Normalize · Tokenize · Vocabulary",
            "field": "Text preprocessing (classic NLP)",
            "why": "Rule-based string handling — no 'learning' happens here yet.",
        },
        {
            "step": "One-hot encoding",
            "field": "Classic Machine Learning",
            "why": "A simple, hand-made way to feed categories into ML models.",
        },
        {
            "step": "Embedding (dense vector)",
            "field": "Deep Learning",
            "why": "In real systems an embedding layer is trained — these vectors are learned, not coded.",
        },
        {
            "step": "Hidden layer  (h = ReLU(W·x + b))",
            "field": "Deep Learning / Neural Networks",
            "why": "This IS a neural-network layer — the literal building block of deep learning.",
        },
        {
            "step": "Co-occurrence grid",
            "field": "Statistical NLP / classic ML",
            "why": "Counting word neighbours is the pre-deep-learning way to capture meaning.",
        },
    ]
