# 🏗️ Architecture

This document explains how **LLM Visualizer** is put together and — most
importantly — **how to add a new module**. The whole design optimises for one
thing: making each AI concept easy to read, run, and *see*.

---

## The two halves

```
┌─────────────────────────┐         HTTP (JSON)        ┌──────────────────────────┐
│   Frontend (the eyes)    │  ───────────────────────►  │   Backend (the brain)    │
│   React + Vite + TS      │                            │   Python + FastAPI       │
│   :5173                  │  ◄───────────────────────  │   :8000                  │
│   draws the JSON         │     structured results     │   computes everything    │
└─────────────────────────┘                            └──────────────────────────┘
```

- The **backend** does all the real computation (tokenizing, vectors, math) and
  returns rich, structured JSON.
- The **frontend** is "dumb" on purpose: it sends your input and **draws**
  whatever the backend returns. No ML logic lives in the UI.

They are fully separate so you can read, run, and understand each on its own.

---

## Request flow (one module)

```
You type text in a page
        │
        ▼
frontend/src/api.ts            ──POST /api/text/<module>──►   backend/app/api/routes.py
        │                                                              │
        │                                                              ▼
        │                                              backend/app/core/<module>.py
        │                                              (pure functions do the work)
        │                                                              │
        ◄────────────────── structured JSON ──────────────────────────┘
        │
        ▼
frontend/src/pages/<Module>.tsx  draws tables / heatmaps / chips
```

---

## Directory layout

```
backend/
├── app/
│   ├── main.py            # FastAPI app, CORS, health route
│   ├── api/routes.py      # ALL endpoints (thin: parse → call core → return)
│   ├── core/              # one file per module — the actual teaching logic
│   │   ├── text_pipeline.py   # Module 1
│   │   ├── tokenization.py    # Module 2
│   │   ├── embeddings.py      # Module 3
│   │   ├── positional.py      # Module 4
│   │   └── concepts.py        # the big-picture / roadmap data
│   └── models/schemas.py  # pydantic request/response shapes
└── requirements.txt

frontend/
└── src/
    ├── api.ts             # one typed function per backend endpoint
    ├── modules.ts         # central registry: nav + routes + roadmap index
    ├── App.tsx            # sidebar + router, both rendered from modules.ts
    ├── pages/             # one page component per module
    └── styles.css         # shared light theme (CSS variables)
```

---

## Conventions (follow these when adding modules)

1. **Functional first.** Each `core/*.py` is a set of small pure functions
   (`normalize → tokenize → vectorize …`), chained by one `run_*()` orchestrator.
   No hidden state, easy to unit-test.

2. **Run from scratch live; show the real library as a snippet.** We compute the
   math ourselves on *your* input so you can see every number, then include a
   static, copyable `code_sample` showing how you'd do it with a real library
   (tiktoken, scikit-learn, PyTorch…).

3. **Data-driven teaching.** Each response carries:
   - `explainers` — "how is this value computed?" strings
   - `field_map` — which area of AI each step belongs to (NLP / ML / DL)
   - `code_sample` — the real-world way
   - `response` — a plain-English summary
   The frontend just renders these; no teaching text is hard-coded in the UI.

4. **Reuse, don't duplicate.** Later modules import helpers from earlier ones
   (e.g. `positional.py` reuses `normalize_text`, `tokenize`, `embedding` from
   `text_pipeline.py`).

5. **Small, readable numbers.** Keep dimensions tiny (4–8) so vectors fit on
   screen.

---

## ➕ How to add a new module (checklist)

Say you're adding **Module 5: Self-Attention**.

### Backend
1. Create `backend/app/core/attention.py` with pure functions + a
   `run_attention(text, ...)` orchestrator returning a dict that includes
   `explainers`, `field_map`, `code_sample`, `response`.
2. Add `AttentionRequest` / `AttentionResponse` to
   `backend/app/models/schemas.py`.
3. Add the route in `backend/app/api/routes.py`:
   ```python
   @router.post("/text/attention", response_model=AttentionResponse)
   def text_attention(req: AttentionRequest):
       return run_attention(req.text)
   ```
4. (Optional) add the path to the list in `main.py` and flip its status to
   `"done"` in `core/concepts.py`.

### Frontend
5. Add a typed client method in `frontend/src/api.ts`:
   ```ts
   attention: (text: string) => postJSON<AttentionResult>("/api/text/attention", { text }),
   ```
6. Create `frontend/src/pages/Attention.tsx` that calls it and draws the result.
7. Register it in `frontend/src/modules.ts` — change its entry from `planned` to
   `done` and attach the `component`. **The sidebar and router update
   automatically.**

That's it — no other files need editing.

---

## Why this scales

Because navigation, routing, and the roadmap index all read from a **single
list** (`modules.ts` on the front end, `concepts.py` on the back end), the
growing curriculum stays consistently numbered and ordered. Adding lesson #11
is the same amount of work as adding #5.
