import { useState } from "react";
import { api, type PositionalResult } from "../api";

/**
 * Module 4 visualizer — Positional Encoding.
 *
 * Shows the sinusoidal PE matrix, adds it to token embeddings, and demonstrates
 * that swapping word order changes the resulting vectors.
 */

// Diverging colour for values in [-1, 1]: blue = positive, red = negative.
function divColor(v: number): string {
  const a = Math.min(Math.abs(v), 1);
  return v >= 0 ? `rgba(37,99,235,${a})` : `rgba(220,38,38,${a})`;
}

export default function Positional() {
  const [text, setText] = useState("the dog bites the man");
  const [dim, setDim] = useState(8);
  const [result, setResult] = useState<PositionalResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  async function run() {
    setLoading(true);
    setErr(null);
    try {
      setResult(await api.positional(text, dim));
    } catch (e) {
      setErr(String(e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div>
      <h2>Module 4 — Positional Encoding</h2>
      <p className="muted">
        Attention (Module 5) reads all words at once, as an unordered set — so it
        can't tell <i>"dog bites man"</i> from <i>"man bites dog"</i>. Positional
        encoding stamps each token with <b>where</b> it sits.
      </p>

      <div className="card">
        <div className="row">
          <input
            type="text"
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && run()}
            placeholder="Type a sentence…"
          />
          <button onClick={run} disabled={loading}>
            {loading ? "Running…" : "Encode positions ▶"}
          </button>
        </div>
        <div className="slider-row">
          <span className="muted">Embedding dimension (d):</span>
          <input
            type="range"
            min={4}
            max={16}
            step={2}
            value={dim}
            onChange={(e) => setDim(Number(e.target.value))}
          />
          <span className="slider-val">{dim}</span>
        </div>
        {err && <div className="note">⚠️ {err}. Is the backend running on port 8000?</div>}
      </div>

      {result && <PositionalView result={result} />}
    </div>
  );
}

function PositionalView({ result }: { result: PositionalResult }) {
  const { explainers, worked_example: w, order_demo: od } = result;

  return (
    <>
      {/* Why */}
      <div className="card">
        <h3>Why positional encoding?</h3>
        <div className="explain">{explainers.why}</div>
      </div>

      {/* PE matrix heatmap */}
      <div className="card">
        <h3>Step 1 · The positional-encoding matrix</h3>
        <div className="explain">{explainers.how}</div>
        <p className="muted">
          Rows = positions in your sentence, columns = the {result.dim} dimensions.
          <span style={{ color: "#2563eb" }}> Blue = positive</span>,{" "}
          <span style={{ color: "#dc2626" }}>red = negative</span>. Notice the
          left columns wiggle fast, the right ones slowly — that mix makes every
          row unique.
        </p>
        <div style={{ overflowX: "auto" }}>
          <table className="heat">
            <thead>
              <tr>
                <th>pos \ dim</th>
                {Array.from({ length: result.dim }, (_, j) => (
                  <th key={j}>d{j}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {result.pe_matrix.map((row, i) => (
                <tr key={i}>
                  <th>{i} ({result.tokens[i]})</th>
                  {row.map((v, j) => (
                    <td key={j} style={{ background: divColor(v), color: Math.abs(v) > 0.6 ? "#fff" : "var(--text)" }}>
                      {v}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Worked example */}
      <div className="card">
        <h3>The formula, worked out (position {w.pos})</h3>
        <div className="vec" style={{ lineHeight: 2 }}>
          <span className="formula">{w.even.formula}</span><br />
          dimension {w.even.dim}: {w.even.calc}<br />
          <br />
          <span className="formula">{w.odd.formula}</span><br />
          dimension {w.odd.dim}: {w.odd.calc}
        </div>
        <div className="explain" style={{ marginTop: "0.8rem" }}>{w.note}</div>
      </div>

      {/* Add to embeddings */}
      <div className="card">
        <h3>Step 2 · embedding + positional = position-aware vector</h3>
        <div className="explain">{explainers.added}</div>
        <div style={{ overflowX: "auto" }}>
          <table>
            <thead>
              <tr>
                <th>pos</th>
                <th>token</th>
                <th>embedding</th>
                <th>+ positional</th>
                <th>= result</th>
              </tr>
            </thead>
            <tbody>
              {result.per_token.map((t) => (
                <tr key={t.position}>
                  <td>{t.position}</td>
                  <td><b>{t.token}</b></td>
                  <td className="vec">[{t.embedding.join(", ")}]</td>
                  <td className="vec">[{t.positional.join(", ")}]</td>
                  <td className="vec">[{t.result.join(", ")}]</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Order demo */}
      <div className="card">
        <h3>Step 3 · Why order matters: same words, different vectors</h3>
        <div className="compare">
          {[od.a, od.b].map((s, idx) => (
            <div key={idx} className="stat" style={{ textAlign: "left" }}>
              <div style={{ fontWeight: 700, marginBottom: "0.5rem" }}>“{s.sentence}”</div>
              {s.vectors.map((v, i) => (
                <div key={i} className="vec" style={{ marginBottom: "0.2rem" }}>
                  <b>{v.token}</b> @ {v.position}: [{v.result.slice(0, 4).join(", ")}…]
                </div>
              ))}
            </div>
          ))}
        </div>
        <div className="note" style={{ marginBottom: 0 }}>🔀 {od.note}</div>
      </div>

      {/* Position similarity */}
      <div className="card">
        <h3>Bonus · How similar are the position vectors?</h3>
        <p className="muted">
          Cosine similarity between each pair of position fingerprints. The
          diagonal is 1.0; nearby positions tend to be more alike.
        </p>
        <div style={{ overflowX: "auto" }}>
          <table className="heat">
            <thead>
              <tr>
                <th>pos × pos</th>
                {result.tokens.map((_, j) => (
                  <th key={j}>p{j}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {result.position_similarity.map((row, i) => (
                <tr key={i}>
                  <th>p{i}</th>
                  {row.map((v, j) => (
                    <td key={j} style={{ background: `rgba(37,99,235,${Math.max(v, 0) * 0.85})`, color: v > 0.6 ? "#fff" : "var(--text)" }}>
                      {v.toFixed(2)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
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
      </div>

      {/* Code */}
      <div className="card">
        <h3>The real-world way (copyable)</h3>
        <pre className="code">{result.code_sample}</pre>
      </div>

      {/* Summary */}
      <div className="card">
        <span className="step-label">Summary</span>
        <p style={{ marginBottom: 0 }}>{result.response}</p>
      </div>
    </>
  );
}
