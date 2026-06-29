# 📚 Curriculum

The learning path, from the absolute basics to **Agentic AI**. Each module is a
single interactive page that runs live on your own input.

## The big picture first

```
AI  ⊃  Machine Learning (ML)  ⊃  Deep Learning  ⊃  LLMs / Transformers  →  Agentic AI
```

- **AI** — making machines act intelligently (the whole field).
- **ML** — a *subset* of AI: machines that learn from data.
- **Deep Learning** — a subset of ML: neural networks with many layers.
- **NLP** — a *domain* of AI for language. It **uses** ML/DL; ML is **not** a part
  of NLP. They overlap.
- **Agentic AI** — an LLM that can plan, use tools, and take actions. The goal.

> ⚠️ Common mistake: *"ML is part of NLP."* It's the other way around — **NLP uses
> ML**, and ML is far bigger (it also powers vision, robotics, recommendations…).

---

## Modules

| # | Module | What you learn | Key math | Status |
|---|--------|----------------|----------|--------|
| 1 | **Text → Response** | How text becomes numbers, vectors, and a response | one-hot, dense vectors, `h = ReLU(W·x + b)`, Cartesian product | ✅ |
| 2 | **Tokenization (BPE)** | How LLMs split text into subword tokens | greedy most-frequent-pair merging | ✅ |
| 3 | **Embeddings & Similarity** | Turning documents into vectors and measuring closeness | TF-IDF, cosine similarity | ✅ |
| 4 | **Positional Encoding** | How a model knows word *order* | sinusoidal `sin/cos(pos / 10000^(2i/d))` | ✅ |
| 5 | Self-Attention (Q/K/V) | The core mechanism of Transformers | scaled dot-product attention | ⏳ |
| 6 | Logits & Softmax | Turning raw scores into probabilities | softmax | ⏳ |
| 7 | Sampling & Generation | How text is actually produced | temperature, top-k / top-p | ⏳ |
| 8 | Using a Real Model | Running GPT-2 locally | — | ⏳ |
| 9 | RAG | Answering from your own documents | retrieval + generation | ⏳ |
| 10 | **Agentic AI** | Tools, planning, multi-step actions | — | ⏳ |

---

## How each module is taught

Every module follows the same shape so the ideas build on each other:

1. **Why** — the problem this concept solves.
2. **The math** — the formula, then the same formula *worked out on your input*.
3. **From scratch** — every number computed live by the Python backend.
4. **Visualised** — tables, heatmaps, chips, bars.
5. **Where it fits** — a map of which AI field (NLP / ML / DL) each step belongs to.
6. **The real-world way** — a copyable snippet using a production library.

---

## Suggested order

Go top to bottom. Each module assumes the previous ones:

- Module 3 (embeddings) builds on Module 1's idea of vectors.
- Module 4 (positional encoding) adds to those embedding vectors.
- Module 5 (attention) will consume the position-aware vectors from Module 4.

Take your time on Modules 1–4 — they're the foundation everything else stands on.
