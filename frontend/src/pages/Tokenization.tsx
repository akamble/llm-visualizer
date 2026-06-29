import { useState } from "react";
import { api, type TokenizationResult } from "../api";

/**
 * Module 2 visualizer — Byte-Pair Encoding (BPE).
 *
 * Two phases, exactly like a real tokenizer:
 *   1. TRAIN merge rules on a fixed corpus (shown step-by-step).
 *   2. ENCODE the user's text using those rules (runs live).
 */
export default function Tokenization() {
  const [text, setText] = useState("slowest learning");
  const [merges, setMerges] = useState(12);
  const [result, setResult] = useState<TokenizationResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  async function run() {
    setLoading(true);
    setErr(null);
    try {
      setResult(await api.tokenize(text, merges));
    } catch (e) {
      setErr(String(e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div>
      <h2>Module 2 — Tokenization with Byte-Pair Encoding (BPE)</h2>
      <p className="muted">
        Real LLMs don't split text into whole words — they use <b>subword</b>{" "}
        tokens learned by an algorithm called BPE. Watch it learn merge rules,
        then tokenize your text.
      </p>

      <div className="card">
        <div className="row">
          <input
            type="text"
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && run()}
            placeholder="Type text to tokenize…"
          />
          <button onClick={run} disabled={loading}>
            {loading ? "Running…" : "Tokenize ▶"}
          </button>
        </div>
        <div className="slider-row">
          <span className="muted">Merge rules to learn:</span>
          <input
            type="range"
            min={1}
            max={30}
            value={merges}
            onChange={(e) => setMerges(Number(e.target.value))}
          />
          <span className="slider-val">{merges}</span>
        </div>
        {err && <div className="note">⚠️ {err}. Is the backend running on port 8000?</div>}
      </div>

      {result && <TokenizationView result={result} />}
    </div>
  );
}

function TokenChips({ tokens }: { tokens: string[] }) {
  return (
    <span className="tokens">
      {tokens.map((t, i) => {
        const eow = t.endsWith("</w>");
        const head = eow ? t.slice(0, -4) : t;
        return (
          <span key={i} className="token">
            {head}
            {eow && <span className="eow">⏎</span>}
          </span>
        );
      })}
    </span>
  );
}

function TokenizationView({ result }: { result: TokenizationResult }) {
  const { training, encoding, comparison, explainers } = result;

  return (
    <>
      {/* Why subword */}
      <div className="card">
        <h3>Why subwords?</h3>
        <div className="explain">{explainers.why_subword}</div>
        <p className="muted" style={{ marginBottom: 0 }}>
          The <code>⏎</code> badge on a token marks the <b>end of a word</b>{" "}
          (BPE's <code>{"</w>"}</code> marker). {explainers.end_of_word}
        </p>
      </div>

      {/* Phase 1 — training corpus */}
      <div className="card">
        <span className="step-label">Phase 1 · Training corpus</span>
        <p className="muted">
          BPE first learns its rules from a corpus (a fixed sample here). Each chip
          is a word and how often it appears:
        </p>
        <div className="chips">
          {Object.entries(result.corpus).map(([w, f]) => (
            <span key={w} className="chip">{w} <b>×{f}</b></span>
          ))}
        </div>
      </div>

      {/* Phase 1 — merge steps */}
      <div className="card">
        <h3>Phase 1 · Learning merge rules, step by step</h3>
        <div className="explain">{explainers.training}</div>
        <div style={{ overflowX: "auto" }}>
          <table>
            <thead>
              <tr>
                <th>#</th>
                <th>most frequent pair</th>
                <th>count</th>
                <th>→ new token</th>
                <th>vocab size</th>
                <th>example word</th>
              </tr>
            </thead>
            <tbody>
              {training.history.map((h) => (
                <tr key={h.step}>
                  <td>{h.step}</td>
                  <td className="vec">({h.pair[0]}, {h.pair[1]})</td>
                  <td>{h.count}</td>
                  <td><span className="token">{h.merged.replace("</w>", "⏎")}</span></td>
                  <td>{h.vocab_size}</td>
                  <td className="vec">
                    {h.example.before.join(" ")} <span className="arrow">→</span>{" "}
                    {h.example.after.join(" ")}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="muted" style={{ marginTop: "0.8rem" }}>
          Notice how the vocabulary <b>grows by one</b> each merge, and frequent
          chunks like <code>lo</code>, <code>low</code>, <code>est⏎</code> emerge.
        </p>
      </div>

      {/* Phase 2 — encoding the user's text */}
      <div className="card">
        <h3>Phase 2 · Encoding your text: “{result.input}”</h3>
        <div className="explain">{explainers.encoding}</div>
        {encoding.per_word.map((pw, i) => (
          <div key={i} className="word-block">
            <span className="src">{pw.word}</span>
            <span className="arrow">→</span>
            <TokenChips tokens={pw.tokens} />
          </div>
        ))}
        <div className="note" style={{ marginBottom: 0 }}>
          💡 Words made of learned chunks (like <code>slowest → s · low · est⏎</code>)
          get few tokens. Unfamiliar words fall back to characters — that's how BPE
          guarantees it can encode <i>anything</i>, with no "unknown word".
        </div>
      </div>

      {/* Comparison */}
      <div className="card">
        <h3>Word vs Character vs BPE</h3>
        <div className="compare">
          <div className="stat">
            <div className="num">{comparison.word_level_vocab}</div>
            <div className="lbl2">word-level vocab<br />(needs every word)</div>
          </div>
          <div className="stat">
            <div className="num">{comparison.char_level_vocab}</div>
            <div className="lbl2">character vocab<br />(tiny, but long sequences)</div>
          </div>
          <div className="stat">
            <div className="num">{comparison.bpe_vocab}</div>
            <div className="lbl2">BPE vocab<br />(chars + {result.num_merges} merges)</div>
          </div>
          <div className="stat">
            <div className="num">{comparison.input_bpe_tokens}</div>
            <div className="lbl2">your text =<br />{comparison.input_bpe_tokens} BPE tokens</div>
          </div>
        </div>
        <p className="muted" style={{ marginTop: "0.8rem", marginBottom: 0 }}>
          Real tokenizers (GPT-2) use ~50,000 merges — big enough to keep
          sequences short, small enough to never run out of vocabulary.
        </p>
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
          ⚠️ Important: BPE itself is <b>not</b> deep learning — it's a deterministic
          algorithm that <i>prepares</i> text. The learning happens in the next
          lessons, when these tokens become vectors fed to a neural network.
        </div>
      </div>

      {/* Real-library code */}
      <div className="card">
        <h3>The real-world way (copyable)</h3>
        <p className="muted">
          We built BPE from scratch to understand it. In practice you'd use a
          battle-tested library:
        </p>
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
