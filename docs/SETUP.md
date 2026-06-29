# 🛠️ Setup Guide

How to run **LLM Visualizer** on your own machine, step by step. Everything runs
**locally** — no accounts, no API keys, no cloud.

---

## 1. Prerequisites

| Tool | Version | Check with | Where to get it |
|------|---------|-----------|-----------------|
| **Python** | 3.10 – 3.14 | `python --version` | <https://www.python.org/downloads/> |
| **Node.js** | 18+ | `node --version` | <https://nodejs.org/> |
| **Git** | any | `git --version` | <https://git-scm.com/> |

You'll run **two processes** at the same time: the Python backend and the
React frontend. Use **two terminal windows**.

---

## 2. Get the code

```bash
git clone <your-repo-url> llm-visualizer
cd llm-visualizer
```

---

## 3. Backend (Python + FastAPI)

```bash
cd backend

# Create an isolated environment so packages don't pollute your system Python.
python -m venv .venv

# Activate it:
#   Windows (PowerShell):
.venv\Scripts\Activate.ps1
#   Windows (cmd):
#   .venv\Scripts\activate.bat
#   macOS / Linux:
#   source .venv/bin/activate

# Install dependencies (fastapi, uvicorn, numpy, pydantic).
pip install -r requirements.txt

# Start the API server (auto-reloads when you edit code).
uvicorn app.main:app --reload --port 8000
```

✅ You should see `Uvicorn running on http://127.0.0.1:8000`.

- API root: <http://localhost:8000/>
- **Interactive API docs** (try every endpoint live): <http://localhost:8000/docs>

Leave this terminal running.

---

## 4. Frontend (React + Vite + TypeScript)

Open a **second terminal**:

```bash
cd frontend

# Install dependencies (first time only).
npm install

# Start the dev server (hot-reloads on save).
npm run dev
```

✅ Open the URL it prints — usually <http://localhost:5173>.

The sidebar lists every module. Click through them in order.

---

## 5. Stopping & restarting

- Stop either server with **Ctrl + C** in its terminal.
- To restart the backend you must re-activate the venv first (step 3), then run
  the `uvicorn` command again.

---

## 6. Troubleshooting

**`pip install` fails on numpy (Python 3.14):**
The `requirements.txt` uses loose version pins (`>=`) so the newest wheels are
picked. If you still hit a build error, upgrade pip first:
`python -m pip install --upgrade pip`, then retry.

**Frontend shows "⚠️ Couldn't reach the backend":**
The backend isn't running, or it's on a different port. Make sure step 3 is
running on port **8000**. The frontend expects `http://localhost:8000` (set in
[`frontend/src/api.ts`](../frontend/src/api.ts)).

**`uvicorn: command not found`:**
Your virtual environment isn't activated. Re-run the activate command from step 3
(you should see `(.venv)` at the start of your prompt).

**Port already in use:**
Something else is using 8000 or 5173. Start on another port, e.g.
`uvicorn app.main:app --reload --port 8010` (then update `BASE_URL` in
`frontend/src/api.ts`), or `npm run dev -- --port 5174`.

**PowerShell: "running scripts is disabled":**
Allow local scripts for your user:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then re-activate the venv.

---

## 7. What next?

- Read [ARCHITECTURE.md](ARCHITECTURE.md) to understand how the project is wired
  and **how to add your own module**.
- Read [CURRICULUM.md](CURRICULUM.md) for the full learning path.
