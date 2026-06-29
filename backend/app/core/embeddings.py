"""
Lesson 3 — Embeddings & Similarity (TF-IDF + cosine).
=====================================================

In Lesson 1 we turned single words into vectors. Now we turn whole **documents**
into vectors, and — the big payoff — we *measure how similar two pieces of text
are*. This is the foundation of search, recommendations, and (later) RAG.

The plan:
    1. TF   — Term Frequency: how often each word appears in a document.
    2. IDF  — Inverse Document Frequency: rare words are more informative than
              common ones ("the" tells you nothing; "transformer" tells you a lot).
    3. TF-IDF = TF × IDF  -> each document becomes a vector of weights.
    4. Cosine similarity — measure the ANGLE between two vectors:

            cos(θ) = (A · B) / (|A| · |B|)

       Two documents pointing the same direction (similar word mix) score ~1;
       unrelated documents score ~0.

We then use this to rank documents against a search query — a tiny search engine.

Everything is plain functions and runs live on the user's documents. The
tokenizer from Lesson 1 is reused (DRY).
"""

from __future__ import annotations

import math
from typing import Dict, List

import numpy as np

from app.core.text_pipeline import normalize_text, tokenize

# A small default corpus: two "animal" docs + two "AI" docs, with overlap so the
# similarity structure is easy to see.
DEFAULT_DOCS: List[str] = [
    "the cat sat on the mat",
    "the dog sat on the log",
    "machine learning models learn from data",
    "deep learning is a kind of machine learning",
]
DEFAULT_QUERY = "learning from data"


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
def build_vocabulary(docs_tokens: List[List[str]]) -> List[str]:
    """Sorted list of every unique term across all documents."""
    vocab = sorted({tok for toks in docs_tokens for tok in toks})
    return vocab


def term_frequencies(docs_tokens: List[List[str]], vocab: List[str]) -> np.ndarray:
    """TF matrix: rows = documents, columns = vocab terms, cells = raw counts."""
    index = {term: i for i, term in enumerate(vocab)}
    tf = np.zeros((len(docs_tokens), len(vocab)))
    for d, toks in enumerate(docs_tokens):
        for tok in toks:
            tf[d][index[tok]] += 1
    return tf


def inverse_document_frequency(tf: np.ndarray) -> np.ndarray:
    """Smoothed IDF per term:  idf = ln((1 + N) / (1 + df)) + 1.

    The +1 smoothing (sklearn's default) avoids divide-by-zero and means a term
    appearing in EVERY document still gets a small non-zero weight — so two
    identical documents correctly score a cosine of exactly 1.0.
    """
    n_docs = tf.shape[0]
    df = np.count_nonzero(tf > 0, axis=0)        # docs containing each term
    idf = np.log((1 + n_docs) / (1 + df)) + 1.0
    return idf


def tfidf_matrix(tf: np.ndarray, idf: np.ndarray) -> np.ndarray:
    """Element-wise TF × IDF — each row is now a document vector."""
    return tf * idf


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """cos(θ) = (A · B) / (|A| |B|). Returns 0 if either vector is all-zero."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def similarity_matrix(vectors: np.ndarray) -> np.ndarray:
    """All pairwise cosine similarities between document vectors."""
    n = vectors.shape[0]
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            mat[i][j] = cosine_similarity(vectors[i], vectors[j])
    return mat


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def run_embeddings(documents: List[str], query: str) -> Dict[str, object]:
    """Vectorize documents with TF-IDF, build a similarity matrix, rank a query."""
    docs = [d for d in documents if d.strip()] or DEFAULT_DOCS
    docs_tokens = [tokenize(normalize_text(d)) for d in docs]

    vocab = build_vocabulary(docs_tokens)
    tf = term_frequencies(docs_tokens, vocab)
    idf = inverse_document_frequency(tf)
    tfidf = tfidf_matrix(tf, idf)

    sim = similarity_matrix(tfidf)

    # --- Query: vectorize with the SAME vocab + idf, then rank documents -----
    q_tokens = tokenize(normalize_text(query))
    q_tf = term_frequencies([q_tokens], vocab)[0]
    q_vec = q_tf * idf
    ranking = sorted(
        (
            {"doc_index": d, "doc": docs[d], "score": round(cosine_similarity(q_vec, tfidf[d]), 3)}
            for d in range(len(docs))
        ),
        key=lambda r: r["score"],
        reverse=True,
    )

    # --- Worked example: the most similar PAIR of distinct documents ---------
    worked = _worked_example(tfidf, sim, docs)

    response = (
        f"I turned {len(docs)} document(s) into TF-IDF vectors over a "
        f"{len(vocab)}-word vocabulary, measured how similar they are, and ranked "
        f'them against your query "{query}". Best match: '
        f'"{ranking[0]["doc"]}" (cosine {ranking[0]["score"]}).'
    )

    return {
        "documents": docs,
        "query": query,
        "vocab": vocab,
        "tf": np.round(tf, 3).tolist(),
        "idf": {term: round(float(idf[i]), 3) for i, term in enumerate(vocab)},
        "tfidf": np.round(tfidf, 3).tolist(),
        "query_vector": np.round(q_vec, 3).tolist(),
        "similarity_matrix": np.round(sim, 3).tolist(),
        "ranking": ranking,
        "worked_example": worked,
        "explainers": _explainers(),
        "field_map": _field_map(),
        "code_sample": _code_sample(),
        "response": response,
    }


def _worked_example(tfidf: np.ndarray, sim: np.ndarray, docs: List[str]) -> Dict[str, object]:
    """Pick the most similar distinct doc-pair and spell out the cosine math."""
    n = len(docs)
    if n < 2:
        return {}
    best_i, best_j, best = 0, 1, -1.0
    for i in range(n):
        for j in range(i + 1, n):
            if sim[i][j] > best:
                best_i, best_j, best = i, j, sim[i][j]

    a, b = tfidf[best_i], tfidf[best_j]
    dot = float(np.dot(a, b))
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    score = cosine_similarity(a, b)
    return {
        "a_index": best_i,
        "b_index": best_j,
        "a_doc": docs[best_i],
        "b_doc": docs[best_j],
        "dot": round(dot, 3),
        "norm_a": round(na, 3),
        "norm_b": round(nb, 3),
        "score": round(score, 3),
        "formula": "cos(θ) = (A · B) / (|A| · |B|)",
    }


# ---------------------------------------------------------------------------
# Teaching metadata
# ---------------------------------------------------------------------------
def _explainers() -> Dict[str, str]:
    return {
        "tfidf": (
            "How each weight is computed: TF (how often a word appears in THIS "
            "document) × IDF (how rare the word is ACROSS all documents). Common "
            "words like 'the' get a low IDF, so they barely count; distinctive "
            "words get a high IDF and dominate the vector. We use smoothed IDF = "
            "ln((1+N)/(1+df)) + 1 so identical documents score exactly 1.0."
        ),
        "cosine": (
            "How similarity is computed: cosine similarity measures the ANGLE "
            "between two vectors, ignoring their length. Dot the two vectors, then "
            "divide by the product of their magnitudes. 1 = same direction "
            "(very similar), 0 = perpendicular (no shared words), so longer "
            "documents aren't unfairly favoured just for being longer."
        ),
        "search": (
            "How the ranking works: the query is turned into a vector with the "
            "SAME vocabulary and IDF weights as the documents, then we sort the "
            "documents by their cosine similarity to that query vector. That is, "
            "in essence, how a classic search engine works."
        ),
    }


def _field_map() -> List[Dict[str, str]]:
    return [
        {
            "step": "TF / IDF / TF-IDF",
            "field": "Classic Machine Learning / NLP",
            "why": "A statistical, hand-designed feature — no neural network needed.",
        },
        {
            "step": "Cosine similarity",
            "field": "Linear algebra (the math under ML)",
            "why": "A geometric measure of closeness used everywhere in ML.",
        },
        {
            "step": "Ranking documents by a query",
            "field": "Information Retrieval (applied NLP)",
            "why": "The same idea powers search engines — and later, RAG (Lesson 9).",
        },
    ]


def _code_sample() -> str:
    return (
        "# The real-world way — scikit-learn:\n"
        "#   pip install scikit-learn\n"
        "from sklearn.feature_extraction.text import TfidfVectorizer\n"
        "from sklearn.metrics.pairwise import cosine_similarity\n"
        "\n"
        "docs = [\n"
        '    "the cat sat on the mat",\n'
        '    "machine learning models learn from data",\n'
        "]\n"
        "vectorizer = TfidfVectorizer()\n"
        "X = vectorizer.fit_transform(docs)      # TF-IDF matrix (sparse)\n"
        "\n"
        "# Rank documents against a query:\n"
        'q = vectorizer.transform(["learning from data"])\n'
        "scores = cosine_similarity(q, X)[0]\n"
        "print(scores)                           # similarity of query to each doc\n"
        "\n"
        "# NOTE: modern systems swap TF-IDF for LEARNED embeddings\n"
        "# (e.g. sentence-transformers) that capture meaning, not just word overlap.\n"
    )
