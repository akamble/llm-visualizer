"""
Lesson 2 — Tokenization with Byte-Pair Encoding (BPE).
======================================================

In Lesson 1 we split text into whole words. Real LLMs (GPT, Claude, Llama)
don't do that — they use **subword** tokens learned by an algorithm called
**Byte-Pair Encoding (BPE)**.

Why not whole words?
    - Vocabulary explodes: English alone has millions of word forms.
    - Out-of-vocabulary problem: a new word like "visualizingly" would be unknown.

Why not single characters?
    - Sequences become very long and each character carries little meaning.

BPE is the sweet spot in between. It starts from characters and repeatedly
**merges the most frequent adjacent pair** into a new token. Common chunks like
"ing", "er", "low" become single tokens, while rare words still break down into
known pieces — so nothing is ever truly "unknown".

This module mirrors real tokenizers in two phases:
    1. TRAIN merge rules on a fixed corpus (done once)   -> `train_bpe`
    2. ENCODE any new text using those rules             -> `encode_word`

Phase 1 is visualised step-by-step; phase 2 runs live on YOUR input.
Everything here is plain functions, no hidden state.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Tuple

# A small, curated AI-themed training corpus: {word: how often it appears}.
# Chosen so shared chunks (low, new, er, est, ing...) emerge clearly as merges.
TRAINING_CORPUS: Dict[str, int] = {
    "low": 5, "lower": 2, "lowest": 2,
    "new": 6, "newer": 3, "newest": 2,
    "slow": 4, "slower": 2,
    "fast": 3, "faster": 2, "fastest": 2,
    "learning": 4, "learner": 2,
}

END = "</w>"          # marks the end of a word (a real BPE convention)
MAX_MERGES = 30       # safety bound on the slider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _init_splits(corpus: Dict[str, int]) -> Dict[str, List[str]]:
    """Start every word as a list of characters + the end-of-word marker.

    "low" -> ["l", "o", "w", "</w>"]
    """
    return {word: list(word) + [END] for word in corpus}


def _count_pairs(splits: Dict[str, List[str]], corpus: Dict[str, int]) -> Counter:
    """Count every adjacent symbol pair, weighted by how often its word occurs."""
    pairs: Counter = Counter()
    for word, freq in corpus.items():
        symbols = splits[word]
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i + 1])] += freq
    return pairs


def _apply_merge(pair: Tuple[str, str], splits: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """Replace every occurrence of `pair` with the single merged symbol."""
    a, b = pair
    merged = a + b
    new_splits: Dict[str, List[str]] = {}
    for word, symbols in splits.items():
        out: List[str] = []
        i = 0
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                out.append(merged)
                i += 2
            else:
                out.append(symbols[i])
                i += 1
        new_splits[word] = out
    return new_splits


# ---------------------------------------------------------------------------
# Phase 1 — TRAIN the merge rules
# ---------------------------------------------------------------------------
def train_bpe(corpus: Dict[str, int], num_merges: int) -> Dict[str, object]:
    """Learn `num_merges` merge rules from the corpus, recording each step.

    Each round: count all adjacent pairs -> pick the most frequent -> merge it.
    """
    splits = _init_splits(corpus)
    base_vocab = sorted({sym for syms in splits.values() for sym in syms})
    vocab = set(base_vocab)

    merges: List[Tuple[str, str]] = []
    history: List[Dict[str, object]] = []

    for step in range(num_merges):
        pairs = _count_pairs(splits, corpus)
        if not pairs:
            break
        # Most frequent pair (ties broken by first seen via Counter ordering).
        best, count = pairs.most_common(1)[0]

        # Pick one word to show as a concrete before/after example.
        example_word = next(
            (w for w, s in splits.items() if _contains_pair(s, best)), None
        )
        before = list(splits[example_word]) if example_word else []

        splits = _apply_merge(best, splits)
        merges.append(best)
        vocab.add(best[0] + best[1])

        history.append(
            {
                "step": step + 1,
                "pair": [best[0], best[1]],
                "merged": best[0] + best[1],
                "count": count,
                "vocab_size": len(vocab),
                "example": {
                    "word": example_word,
                    "before": before,
                    "after": list(splits[example_word]) if example_word else [],
                },
            }
        )

    return {
        "merges": merges,
        "history": history,
        "base_vocab": base_vocab,
        "final_vocab": sorted(vocab),
    }


def _contains_pair(symbols: List[str], pair: Tuple[str, str]) -> bool:
    a, b = pair
    return any(symbols[i] == a and symbols[i + 1] == b for i in range(len(symbols) - 1))


# ---------------------------------------------------------------------------
# Phase 2 — ENCODE new text with the learned rules
# ---------------------------------------------------------------------------
def encode_word(word: str, merges: List[Tuple[str, str]]) -> List[str]:
    """Tokenize a single word by applying the learned merges in order."""
    symbols = list(word) + [END]
    for a, b in merges:
        merged = a + b
        out: List[str] = []
        i = 0
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                out.append(merged)
                i += 2
            else:
                out.append(symbols[i])
                i += 1
        symbols = out
    return symbols


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def run_tokenization(text: str, num_merges: int = 12) -> Dict[str, object]:
    """Train BPE on the corpus, then encode the user's text and package it all."""
    num_merges = max(1, min(MAX_MERGES, int(num_merges)))

    trained = train_bpe(TRAINING_CORPUS, num_merges)
    merges: List[Tuple[str, str]] = [tuple(m) for m in trained["merges"]]

    # Encode the user's input, word by word.
    words = text.lower().split()
    per_word = [{"word": w, "tokens": encode_word(w, merges)} for w in words]
    all_tokens = [tok for pw in per_word for tok in pw["tokens"]]

    # Side-by-side comparison of vocabulary strategies.
    char_vocab = sorted({c for w in TRAINING_CORPUS for c in w} | {END})
    comparison = {
        "word_level_vocab": len(TRAINING_CORPUS),
        "char_level_vocab": len(char_vocab),
        "bpe_vocab": len(trained["final_vocab"]),
        "input_word_tokens": len(words),
        "input_bpe_tokens": len(all_tokens),
    }

    response = (
        f"I trained {len(merges)} merge rule(s) on a {len(TRAINING_CORPUS)}-word "
        f"corpus, then used them to split your text into {len(all_tokens)} "
        f"subword token(s)."
    )

    return {
        "input": text,
        "num_merges": num_merges,
        "corpus": TRAINING_CORPUS,
        "training": trained,
        "encoding": {"per_word": per_word, "tokens": all_tokens},
        "comparison": comparison,
        "explainers": _explainers(),
        "field_map": _field_map(),
        "code_sample": _code_sample(),
        "response": response,
    }


# ---------------------------------------------------------------------------
# Teaching metadata
# ---------------------------------------------------------------------------
def _explainers() -> Dict[str, str]:
    return {
        "why_subword": (
            "Whole-word vocabularies are huge and choke on unseen words; "
            "single characters make sequences long and meaningless. Subword "
            "tokens are the middle ground: frequent chunks (like 'ing' or 'low') "
            "become one token, while rare words still break into known pieces — "
            "so the model never meets a truly 'unknown' word."
        ),
        "training": (
            "How a merge is chosen: in each round we count every adjacent pair "
            "of symbols across the whole corpus (weighted by word frequency) and "
            "merge the single MOST frequent pair into a new token. Repeat. Greedy "
            "and deterministic — no neural network involved here."
        ),
        "encoding": (
            "How your text is tokenized: each word starts as characters + the "
            "end marker, then we apply the learned merges IN THE ORDER they were "
            "discovered. Whatever pieces remain are the final tokens."
        ),
        "end_of_word": (
            "The </w> marker means 'end of word'. It lets BPE tell apart a chunk "
            "in the middle of a word from the same chunk at the end (e.g. 'er' "
            "inside 'verb' vs 'er</w>' ending 'lower')."
        ),
    }


def _field_map() -> List[Dict[str, str]]:
    return [
        {
            "step": "Word-frequency counting",
            "field": "Text preprocessing (classic NLP)",
            "why": "Just counting — the raw material BPE learns from.",
        },
        {
            "step": "Learning merge rules (BPE training)",
            "field": "Statistical NLP / unsupervised learning",
            "why": "A greedy, data-driven algorithm — but NOT deep learning; no weights are trained.",
        },
        {
            "step": "Encoding text into subword tokens",
            "field": "NLP preprocessing for Deep Learning",
            "why": "Turns text into the token IDs that the neural network (Lessons 3+) will consume.",
        },
    ]


def _code_sample() -> str:
    """The real-library way (shown as a static, copyable snippet)."""
    return (
        "# The real-world way — using OpenAI's tiktoken (GPT-2/4 BPE):\n"
        "#   pip install tiktoken\n"
        "import tiktoken\n"
        "\n"
        'enc = tiktoken.get_encoding("gpt2")\n'
        'ids = enc.encode("I love tokenization")\n'
        "print(ids)                       # -> [40, 1842, 11241, 1634]\n"
        "print([enc.decode([i]) for i in ids])  # -> ['I', ' love', ' token', 'ization']\n"
        "\n"
        "# Or train your own BPE with Hugging Face 'tokenizers':\n"
        "#   pip install tokenizers\n"
        "from tokenizers import Tokenizer, models, trainers, pre_tokenizers\n"
        "tok = Tokenizer(models.BPE())\n"
        "tok.pre_tokenizer = pre_tokenizers.Whitespace()\n"
        "trainer = trainers.BpeTrainer(vocab_size=5000)\n"
        'tok.train(["corpus.txt"], trainer)\n'
        'print(tok.encode("lowest").tokens)\n'
    )
