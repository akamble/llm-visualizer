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
│   │   ├── api/routes.py    # HTTP endpoints (one per module)
│   │   ├── core/            # one file per module — the teaching logic
│   │   │   ├── text_pipeline.py  # Module 1: text → numbers → vectors → weights
│   │   │   ├── tokenization.py   # Module 2: Byte-Pair Encoding
│   │   │   ├── embeddings.py      # Module 3: TF-IDF + cosine similarity
│   │   │   ├── positional.py      # Module 4: positional encoding
│   │   │   └── concepts.py        # The AI/ML/NLP "big picture" data
│   │   └── models/schemas.py     # Request/response shapes (pydantic)
│   └── requirements.txt
│
├── frontend/                # React + Vite + TypeScript — the "eyes"
│   ├── src/
│   │   ├── modules.ts             # central registry: nav + routes + index
│   │   ├── App.tsx                # sidebar + router (rendered from modules.ts)
│   │   ├── api.ts                 # one typed call per backend endpoint
│   │   └── pages/                 # one page component per module
│   └── package.json
│
├── docs/                    # SETUP, ARCHITECTURE, CURRICULUM guides
└── README.md
```

**Why this layout?** Frontend and backend are fully separate so you can run,
read, and understand each on its own. Inside the backend, every learning topic
is its own small file exposing plain functions; on the front end, navigation and
routing are driven by one central [`modules.ts`](frontend/src/modules.ts) index —
so adding a module is a one-entry change. See
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full how-to.

---

## 📖 Documentation

- **[docs/SETUP.md](docs/SETUP.md)** — detailed local setup + troubleshooting
- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** — how it's wired + how to add a module
- **[docs/CURRICULUM.md](docs/CURRICULUM.md)** — the full learning path

---

## 🚀 Quick start (everything runs locally)

You need **Python 3.10+** and **Node.js 18+** installed. (Full guide with
troubleshooting: [docs/SETUP.md](docs/SETUP.md).)

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

## 📚 Module 1: How text becomes a response

The first module answers the question — *"how is text converted and a response
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

1. ✅ **Text Fundamentals** — text → numbers → vectors → weights
2. ✅ **Tokenization** — subword tokens via Byte-Pair Encoding (BPE)
3. ✅ **Embeddings & Similarity** — TF-IDF vectors, cosine similarity, query ranking
4. ✅ **Positional Encoding** — sinusoidal position vectors added to embeddings
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
- **Local & free** — no API keys, no cloud. Modules 1–4 need only `numpy`.
- **Visual** — the backend returns structured JSON; the frontend draws it.

## 📄 License

See [LICENSE](LICENSE).
