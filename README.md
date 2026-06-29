# 🧠 LLM Visualizer — Learn AI from Basics to Agentic AI

An interactive, **local-first** learning project that takes you from the absolute
basics (how text becomes numbers, what a vector is, how a "hidden layer" works)
all the way up to **Agentic AI**.

You learn by *seeing*: every concept is traced through a real pipeline and drawn
on screen. The codebase is intentionally **modular** and written in a
**functional style** so each piece is easy to read, test, and extend.

> **The big picture (read this first):**
>
> ```
> AI  ⊃  Machine Learning (ML)  ⊃  Deep Learning  ⊃  LLMs / Transformers  →  Agentic AI
> ```
>
> - **AI** = the whole field of making machines act intelligently.
> - **ML** = a *subset* of AI: machines that **learn from data**.
> - **Deep Learning** = a subset of ML: neural networks with many layers.
> - **NLP** (Natural Language Processing) = a **domain** of AI for understanding
>   language. NLP **uses** ML/DL — ML is *not* a part of NLP. They overlap.
> - **Agentic AI** = an LLM that can **plan, use tools, and take actions** — our
>   final goal.

---

## 📁 Project structure

```
llm-visualizer/
├── backend/                 # Python (FastAPI) — the "brain"
│   ├── app/
│   │   ├── main.py          # FastAPI app + CORS
│   │   ├── api/routes.py    # HTTP endpoints
│   │   ├── core/
│   │   │   ├── text_pipeline.py  # Lesson 1: text → numbers → vectors → weights
│   │   │   └── concepts.py       # The AI/ML/NLP "big picture" data
│   │   └── models/schemas.py     # Request/response shapes (pydantic)
│   └── requirements.txt
│
├── frontend/                # React + Vite + TypeScript — the "eyes"
│   ├── src/
│   │   ├── pages/BigPicture.tsx    # The AI ⊃ ML ⊃ DL diagram
│   │   ├── pages/TextPipeline.tsx  # Lesson 1 visualizer
│   │   ├── api.ts                  # talks to the backend
│   │   └── App.tsx
│   └── package.json
│
└── README.md
```

**Why this layout?** Frontend and backend are fully separate so you can run,
read, and understand each on its own. Inside the backend, every learning topic
is its own small module exposing plain functions — add `core/module_02_xxx.py`
later without touching anything else.

---

## 🚀 Quick start (everything runs locally)

You need **Python 3.10+** and **Node.js 18+** installed.

### 1. Backend (Python / FastAPI)

```bash
cd backend
python -m venv .venv

# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# macOS / Linux:
# source .venv/bin/activate

pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Backend is now at **http://localhost:8000**.
Interactive API docs (try it live): **http://localhost:8000/docs**

### 2. Frontend (React / Vite)

Open a **second terminal**:

```bash
cd frontend
npm install
npm run dev
```

Frontend is now at **http://localhost:5173**. Open it in your browser.

---

## 📚 Lesson 1: How text becomes a response

The first lesson answers your question — *"how is text converted and a response
returned?"* — by walking one sentence through every step:

| Step | What happens | Concept you learn |
|------|--------------|-------------------|
| 1. Normalize | lowercase, trim, clean | preprocessing |
| 2. Tokenize | split into tokens (words) | tokens |
| 3. Vocabulary | each unique token → an integer ID | text → numbers |
| 4. One-hot | each token → a sparse 0/1 vector | **sparse vectors** |
| 5. Embedding | each token → a small dense vector | **dense vectors** |
| 6. Hidden layer | `h = activation(W · x + b)` | **weights, matrix math, hidden layers** |
| 7. Co-occurrence | vocab × vocab grid | **Cartesian product** |
| 8. Response | a readable summary | how output is formed |

Every number you see in the UI is **computed live** by the backend from *your*
input — nothing is faked.

---

## 🗺️ Roadmap (basic → advanced → agentic)

1. ✅ **Text Fundamentals** — text → numbers → vectors → weights *(this module)*
2. ⏳ Tokenization (subword / BPE)
3. ⏳ Embeddings & similarity (cosine, TF-IDF)
4. ⏳ Positional encoding
5. ⏳ Self-attention (Q / K / V)
6. ⏳ Logits & softmax (turning numbers into probabilities)
7. ⏳ Sampling & text generation (temperature)
8. ⏳ Using a real model (GPT-2 locally)
9. ⏳ Retrieval-Augmented Generation (RAG)
10. ⏳ **Agentic AI** — tools, planning, multi-step actions

Each new module = one new `backend/app/core/module_*.py` + one frontend page.

---

## 🧩 Design principles

- **Functional first** — small pure functions (`normalize → tokenize → vectorize`),
  easy to follow and unit-test.
- **Modular** — one concept per file; register new modules without edits elsewhere.
- **Local & free** — no API keys, no cloud. Lesson 1 needs only `numpy`.
- **Visual** — the backend returns structured JSON; the frontend draws it.

## 📄 License

See [LICENSE](LICENSE).
