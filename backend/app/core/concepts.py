"""
The "big picture" data: how AI, ML, Deep Learning, NLP and Agentic AI relate.

This is served to the frontend so the diagram is driven by data, not hardcoded
in the UI. IMPORTANT conceptual note:

    AI  ⊃  ML  ⊃  Deep Learning  ⊃  LLMs / Transformers
    NLP is a *domain* of AI that USES ML/DL — ML is NOT a part of NLP.

So the common phrasing "ML is part of NLP" is backwards: NLP relies on ML, but
ML is a much bigger field (it also covers vision, recommendations, robotics...).
"""

from typing import Dict, List


def get_hierarchy() -> Dict[str, object]:
    """Return the nested fields of AI as concentric layers (outer -> inner)."""
    return {
        "layers": [
            {
                "name": "Artificial Intelligence (AI)",
                "short": "AI",
                "desc": "Any technique that makes machines act intelligently.",
                "examples": ["rule-based systems", "search", "planning"],
            },
            {
                "name": "Machine Learning (ML)",
                "short": "ML",
                "desc": "A subset of AI: systems that learn patterns from data "
                        "instead of being explicitly programmed.",
                "examples": ["spam filters", "recommendations", "fraud detection"],
            },
            {
                "name": "Deep Learning",
                "short": "DL",
                "desc": "A subset of ML using neural networks with many layers.",
                "examples": ["image recognition", "speech-to-text"],
            },
            {
                "name": "LLMs / Transformers",
                "short": "LLM",
                "desc": "Deep networks trained on huge text corpora to predict "
                        "the next token. The engine behind ChatGPT, Claude, etc.",
                "examples": ["GPT", "Claude", "Llama"],
            },
        ],
        # NLP is drawn as an overlapping domain, not a nesting ring.
        "domain": {
            "name": "Natural Language Processing (NLP)",
            "short": "NLP",
            "desc": "The field of getting computers to understand human language. "
                    "It USES ML/DL — it is not contained by them, and ML is not "
                    "contained by NLP. They overlap.",
            "examples": ["translation", "sentiment analysis", "chatbots"],
        },
        # Our final destination.
        "goal": {
            "name": "Agentic AI",
            "short": "Agent",
            "desc": "An LLM that can plan, call tools, and take multi-step actions "
                    "to accomplish goals — the final stop on this learning path.",
        },
        "note": "Correct nesting: AI ⊃ ML ⊃ Deep Learning ⊃ LLMs. "
                "NLP is a language domain that draws from ML/DL.",
    }


def get_roadmap() -> List[Dict[str, object]]:
    """The basic -> advanced -> agentic learning path."""
    modules = [
        ("Text Fundamentals", "text → numbers → vectors → weights", "done"),
        ("Tokenization", "splitting text into subword tokens (BPE)", "planned"),
        ("Embeddings & Similarity", "dense vectors, cosine, TF-IDF", "planned"),
        ("Positional Encoding", "giving the model a sense of word order", "planned"),
        ("Self-Attention", "Query / Key / Value, the heart of Transformers", "planned"),
        ("Logits & Softmax", "turning raw scores into probabilities", "planned"),
        ("Sampling & Generation", "temperature, top-k, producing text", "planned"),
        ("Using a Real Model", "running GPT-2 locally", "planned"),
        ("RAG", "retrieval-augmented generation with your own docs", "planned"),
        ("Agentic AI", "tools, planning, multi-step actions", "planned"),
    ]
    return [
        {"step": i + 1, "title": t, "desc": d, "status": s}
        for i, (t, d, s) in enumerate(modules)
    ]
