import { useState } from "react";
import { api, type PipelineResult, type TokenRow } from "../api";

/**
 * Lesson 1 visualizer. Sends text to the backend and renders every step of the
 * conversion: normalize -> tokenize -> vocab (numbers) -> one-hot -> embedding
 * -> hidden layer (weights) -> co-occurrence grid (Cartesian product) -> response.
 */
export default function TextPipeline() {
  const [text, setText] = useState("I love AI and AI loves data");
  const [result, setResult] = useState<PipelineResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  async function run() {
    setLoading(true);
    setErr(null);
    try {
      setResult(await api.pipeline(text));
    } catch (e) {
      setErr(String(e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div>
      <h2>Lesson 1 — How text becomes a response</h2>
      <p className="muted">
        Type a sentence and watch it turn into numbers, then vectors, then pass
        through a neural-network layer. Every number is computed live by the
        Python backend from your input.
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
            {loading ? "Running…" : "Run pipeline ▶"}
          </button>
        </div>
        {err && (
          <div className="note">
            ⚠️ {err}. Is the backend running on port 8000?
          </div>
        )}
      </div>

      {result && <PipelineView result={result} />}
    </div>
  );
}

function PipelineView({ result }: { result: PipelineResult }) {
  const s = result.steps;
  const tokens = s["4_to_6_tokens"];

  return (
    <>
      {/* Step 1 + 2 */}
      <div className="card">
        <div className="step">
          <span className="step-label">Step 1 · Normalize</span>
          <div className="vec">"{result.input}" → "{s["1_normalized"]}"</div>
          <div className="muted">lowercased, punctuation removed, spaces collapsed</div>
        </div>
        <div className="step">
          <span className="step-label">Step 2 · Tokenize</span>
          <div className="chips">
            {s["2_tokens"].map((t, i) => (
              <span key={i} className="chip">{t}</span>
            ))}
          </div>
          <div className="muted">{s["2_tokens"].length} tokens (the units the model works with)</div>
        </div>
        <div className="step">
          <span className="step-label">Step 3 · Vocabulary (text → numbers!)</span>
          <div className="chips">
            {Object.entries(s["3_vocab"]).map(([word, id]) => (
              <span key={word} className="chip">{word} → <b>{id}</b></span>
            ))}
          </div>
          <div className="muted">
            Each unique word gets an integer ID. This is the exact moment text
            becomes something a computer can multiply.
          </div>
        </div>
      </div>

      {/* Steps 4-6: per-token table */}
      <div className="card">
        <h3>Steps 4–6 · Each token's journey</h3>
        <p className="muted">
          <b>One-hot</b> = sparse (mostly zeros). <b>Embedding</b> = dense (packed
          meaning, {result.config.embedding_dim} numbers). <b>Hidden</b> = output
          of the neural layer ({result.config.hidden_dim} numbers).
        </p>
        <div style={{ overflowX: "auto" }}>
          <table>
            <thead>
              <tr>
                <th>token</th>
                <th>id</th>
                <th>one-hot (sparse)</th>
                <th>embedding (dense)</th>
                <th>hidden output</th>
              </tr>
            </thead>
            <tbody>
              {tokens.map((t, i) => (
                <tr key={i}>
                  <td><b>{t.token}</b></td>
                  <td>{t.id}</td>
                  <td className="vec">[{t.one_hot.join(", ")}]</td>
                  <td className="vec">[{t.embedding.join(", ")}]</td>
                  <td className="vec">[{t.hidden.output.join(", ")}]</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="explain">
          <b>How is the embedding computed?</b> {result.explainers.embedding}
        </div>
      </div>

      {/* The hidden-layer math, shown for the first token */}
      {tokens.length > 0 && (
        <HiddenLayerMath row={tokens[0]} explainer={result.explainers.hidden} />
      )}

      {/* Step 7: co-occurrence heatmap (Cartesian product) */}
      <div className="card">
        <h3>Step 7 · Co-occurrence grid — a Cartesian product</h3>
        <p className="muted">
          {s["7_cooccurrence"].grid_size}. Each cell counts how often the row word
          was immediately followed by the column word.
        </p>
        <div style={{ overflowX: "auto" }}>
          <table className="heat">
            <thead>
              <tr>
                <th>↓ row → col</th>
                {s["7_cooccurrence"].labels.map((l) => (
                  <th key={l}>{l}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {s["7_cooccurrence"].matrix.map((row, i) => (
                <tr key={i}>
                  <th>{s["7_cooccurrence"].labels[i]}</th>
                  {row.map((v, j) => (
                    <td
                      key={j}
                      style={{
                        background: v > 0 ? `rgba(37,99,235,${0.12 + v * 0.18})` : "transparent",
                      }}
                    >
                      {v}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="explain">
          <b>Why are some cells (like “ai → ai”) zero?</b> {result.explainers.cooccurrence}
        </div>
      </div>

      {/* Where does each step happen? — connects the lesson to the big picture */}
      <div className="card">
        <h3>🗺️ Where does each step happen?</h3>
        <p className="muted">
          Connecting this lesson back to the big picture: each step lives in a
          different area of AI.
        </p>
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

      {/* Step 8: response */}
      <div className="card">
        <span className="step-label">Step 8 · Response</span>
        <p style={{ marginBottom: 0 }}>{result.response}</p>
      </div>
    </>
  );
}

function HiddenLayerMath({ row, explainer }: { row: TokenRow; explainer: string }) {
  const h = row.hidden;
  const x = row.embedding;
  const w0 = h.weights[0]; // first row of W -> first hidden number

  // Build the worked-out arithmetic for the FIRST hidden neuron, live.
  const terms = w0.map((w, i) => `(${w} × ${x[i]})`).join(" + ");
  const products = w0.reduce((sum, w, i) => sum + w * x[i], 0);
  const z0 = h.pre_activation[0];
  const h0 = h.output[0];

  return (
    <div className="card">
      <h3>The math inside a hidden layer (token "{row.token}")</h3>
      <p>
        The single most important formula in deep learning:&nbsp;
        <span className="formula">{h.formula}</span>
      </p>
      <ul className="muted">
        <li><b>x</b> = the embedding (input): <span className="vec">[{row.embedding.join(", ")}]</span></li>
        <li><b>W</b> = weight matrix ({h.weights.length}×{h.weights[0].length}) — what the network "knows"</li>
        <li><b>b</b> = bias: <span className="vec">[{h.bias.join(", ")}]</span></li>
        <li><b>W · x + b</b> (pre-activation z): <span className="vec">[{h.pre_activation.join(", ")}]</span></li>
        <li><b>ReLU(z)</b> = max(0, z) (output h): <span className="vec">[{h.output.join(", ")}]</span></li>
      </ul>

      <div className="explain">
        <b>How is the hidden output computed?</b> {explainer}
      </div>

      {/* Live worked example for the first hidden neuron */}
      <div className="step">
        <span className="step-label">Worked example · first hidden number</span>
        <div className="vec" style={{ lineHeight: 1.9 }}>
          z₀ = (W row 0 · x) + b₀<br />
          z₀ = {terms} + ({h.bias[0]})<br />
          z₀ = {products.toFixed(3)} + ({h.bias[0]}) = <b>{z0}</b><br />
          h₀ = ReLU({z0}) = max(0, {z0}) = <b>{h0}</b>
        </div>
        <div className="muted" style={{ marginTop: "0.4rem" }}>
          The same recipe runs for each of the {h.output.length} hidden numbers,
          each using a different row of W.
        </div>
      </div>

      <details>
        <summary className="muted">Show weight matrix W</summary>
        <table style={{ marginTop: "0.5rem", maxWidth: 400 }}>
          <tbody>
            {h.weights.map((wrow, i) => (
              <tr key={i}>
                {wrow.map((w, j) => (
                  <td key={j} className="vec">{w}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </details>
      <div className="note">
        💡 Notice each output number mixes <i>all</i> input numbers — that's what
        matrix multiplication does. Stack many such layers and you get a{" "}
        <b>deep</b> network.
      </div>
    </div>
  );
}
