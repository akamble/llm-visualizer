import { useState } from "react";
import { api, type EmbeddingsResult } from "../api";

/**
 * Lesson 3 visualizer — Embeddings & Similarity.
 *
 * Turns documents into TF-IDF vectors, measures pairwise cosine similarity, and
 * ranks the documents against a search query (a tiny search engine).
 */
const DEFAULT_DOCS = [
  "the cat sat on the mat",
  "the dog sat on the log",
  "machine learning models learn from data",
  "deep learning is a kind of machine learning",
].join("\n");

export default function Embeddings() {
  const [docsText, setDocsText] = useState(DEFAULT_DOCS);
  const [query, setQuery] = useState("learning from data");
  const [result, setResult] = useState<EmbeddingsResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  async function run() {
    setLoading(true);
    setErr(null);
    try {
      const docs = docsText.split("\n").map((d) => d.trim()).filter(Boolean);
      setResult(await api.embeddings(docs, query));
    } catch (e) {
      setErr(String(e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div>
      <h2>Lesson 3 — Embeddings &amp; Similarity (TF-IDF + cosine)</h2>
      <p className="muted">
        Now we turn whole <b>documents</b> into vectors and measure how similar
        they are — the foundation of search, recommendations, and (later) RAG.
      </p>

      <div className="card">
        <span className="step-label">Documents (one per line)</span>
        <textarea value={docsText} onChange={(e) => setDocsText(e.target.value)} />
        <div className="slider-row" style={{ marginTop: "0.8rem" }}>
          <span className="muted">Search query:</span>
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && run()}
          />
          <button onClick={run} disabled={loading}>
            {loading ? "Running…" : "Analyze ▶"}
          </button>
        </div>
        {err && <div className="note">⚠️ {err}. Is the backend running on port 8000?</div>}
      </div>

      {result && <EmbeddingsView result={result} />}
    </div>
  );
}

function EmbeddingsView({ result }: { result: EmbeddingsResult }) {
  const { explainers, worked_example: w } = result;
  const maxScore = Math.max(...result.ranking.map((r) => r.score), 0.0001);

  return (
    <>
      {/* Step 1: TF-IDF vectors */}
      <div className="card">
        <h3>Step 1 · TF-IDF vectors</h3>
        <div className="explain">{explainers.tfidf}</div>
        <p className="muted">
          Each document becomes a row of weights over the {result.vocab.length}-word
          vocabulary. Bigger weight = the word is frequent here <i>and</i> rare overall.
        </p>
        <div style={{ overflowX: "auto" }}>
          <table className="heat">
            <thead>
              <tr>
                <th>doc \ term</th>
                {result.vocab.map((t) => (
                  <th key={t}>
                    {t}
                    <div className="muted" style={{ fontSize: "0.7rem", fontWeight: 400 }}>
                      idf {result.idf[t]}
                    </div>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {result.tfidf.map((row, i) => (
                <tr key={i}>
                  <th style={{ textAlign: "left", maxWidth: 180 }}>{result.documents[i]}</th>
                  {row.map((v, j) => (
                    <td
                      key={j}
                      style={{ background: v > 0 ? `rgba(8,145,178,${0.1 + Math.min(v / 4, 0.7)})` : "transparent" }}
                    >
                      {v || ""}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Step 2: similarity matrix */}
      <div className="card">
        <h3>Step 2 · Cosine similarity between documents</h3>
        <div className="explain">{explainers.cosine}</div>
        <div style={{ overflowX: "auto" }}>
          <table className="heat">
            <thead>
              <tr>
                <th>↓ doc × doc →</th>
                {result.documents.map((_, j) => (
                  <th key={j}>D{j}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {result.similarity_matrix.map((row, i) => (
                <tr key={i}>
                  <th style={{ textAlign: "left" }}>D{i}: {result.documents[i]}</th>
                  {row.map((v, j) => (
                    <td key={j} style={{ background: `rgba(37,99,235,${v * 0.85})`, color: v > 0.6 ? "#fff" : "var(--text)" }}>
                      {v.toFixed(2)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="muted" style={{ marginTop: "0.7rem", marginBottom: 0 }}>
          The diagonal is always 1.0 (a document is identical to itself). Brighter
          off-diagonal cells = more similar documents.
        </p>
      </div>

      {/* Worked example */}
      {w && w.formula && (
        <div className="card">
          <h3>The cosine formula, worked out</h3>
          <p>
            Most similar pair: <b>D{w.a_index}</b> (“{w.a_doc}”) and <b>D{w.b_index}</b> (“{w.b_doc}”).
            &nbsp;<span className="formula">{w.formula}</span>
          </p>
          <div className="vec" style={{ lineHeight: 2 }}>
            A · B (dot product) = <b>{w.dot}</b><br />
            |A| (length of A) = {w.norm_a}, &nbsp; |B| (length of B) = {w.norm_b}<br />
            cos(θ) = {w.dot} / ({w.norm_a} × {w.norm_b}) = <b>{w.score}</b>
          </div>
        </div>
      )}

      {/* Step 3: query ranking — the search engine */}
      <div className="card">
        <h3>Step 3 · Ranking documents by your query: “{result.query}”</h3>
        <div className="explain">{explainers.search}</div>
        {result.ranking.map((r) => (
          <div key={r.doc_index} className="rank-row">
            <div className="rank-bar-track">
              <div className="rank-bar-fill" style={{ width: `${(r.score / maxScore) * 100}%` }} />
              <span className="rank-bar-label">{r.doc}</span>
            </div>
            <div className="rank-score">{r.score}</div>
          </div>
        ))}
        <div className="note" style={{ marginBottom: 0 }}>
          🔎 This is, in miniature, how a search engine ranks results — by vector
          similarity to your query.
        </div>
      </div>

      {/* Field map */}
      <div className="card">
        <h3>🗺️ Where does each step happen?</h3>
        <div className="fieldmap">
          {result.field_map.map((f, i) => (
            <div key={i} className="fieldmap-row">
              <span><b>{f.step}</b></span>
              <span className="badge">{f.field}</span>
              <span className="why">{f.why}</span>
            </div>
          ))}
        </div>
        <div className="note">
          💡 TF-IDF only counts word <i>overlap</i> — it thinks "car" and
          "automobile" are unrelated. In later lessons, <b>learned embeddings</b>{" "}
          fix this by capturing <i>meaning</i>, so synonyms land near each other.
        </div>
      </div>

      {/* Real-library code */}
      <div className="card">
        <h3>The real-world way (copyable)</h3>
        <pre className="code">{result.code_sample}</pre>
      </div>

      {/* Response */}
      <div className="card">
        <span className="step-label">Summary</span>
        <p style={{ marginBottom: 0 }}>{result.response}</p>
      </div>
    </>
  );
}
