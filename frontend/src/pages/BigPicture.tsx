import { useEffect, useState } from "react";
import { api, type Hierarchy } from "../api";

/**
 * The "big picture" page. Draws the CORRECT nesting of AI fields as concentric
 * rings — AI ⊃ ML ⊃ Deep Learning ⊃ LLMs — with NLP shown as an overlapping
 * domain (because NLP *uses* ML; ML is not inside NLP).
 */
export default function BigPicture() {
  const [data, setData] = useState<Hierarchy | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    api.hierarchy().then(setData).catch((e) => setErr(String(e)));
  }, []);

  return (
    <div>
      <h2>The Big Picture: where does it all fit?</h2>
      <p className="muted">
        Before touching any model, get the map right. These fields are{" "}
        <b>nested</b>, not separate boxes.
      </p>

      {err && (
        <div className="note">
          ⚠️ Couldn't reach the backend ({err}). Start it with{" "}
          <code>uvicorn app.main:app --reload --port 8000</code> in the{" "}
          <code>backend/</code> folder.
        </div>
      )}

      <div className="card">
        {/* Concentric rings drawn with plain SVG — no chart library needed. */}
        <svg viewBox="0 0 520 360" width="100%" style={{ maxWidth: 560 }}>
          <defs>
            <style>{`
              .ring { fill-opacity: 1; stroke-width: 2; }
              .lbl { font: 600 13px system-ui; fill: #1e293b; }
              .sub { font: 11px system-ui; fill: #64748b; }
            `}</style>
          </defs>

          {/* AI — outer */}
          <ellipse cx="230" cy="180" rx="220" ry="160" fill="#eff6ff" className="ring" stroke="#2563eb" />
          <text x="230" y="40" textAnchor="middle" className="lbl">Artificial Intelligence (AI)</text>

          {/* ML */}
          <ellipse cx="210" cy="190" rx="160" ry="120" fill="#ecfeff" className="ring" stroke="#0891b2" />
          <text x="210" y="92" textAnchor="middle" className="lbl">Machine Learning</text>

          {/* Deep Learning */}
          <ellipse cx="200" cy="200" rx="105" ry="80" fill="#fefce8" className="ring" stroke="#ca8a04" />
          <text x="200" y="138" textAnchor="middle" className="lbl">Deep Learning</text>

          {/* LLMs — inner */}
          <ellipse cx="195" cy="210" rx="58" ry="42" fill="#fdf2f8" className="ring" stroke="#db2777" />
          <text x="195" y="214" textAnchor="middle" className="lbl">LLMs</text>

          {/* NLP — overlapping domain on the right, deliberately straddling AI/ML */}
          <ellipse cx="400" cy="230" rx="95" ry="70" fill="rgba(147,51,234,0.10)" className="ring" stroke="#9333ea" strokeDasharray="5 4" />
          <text x="425" y="225" textAnchor="middle" className="lbl">NLP</text>
          <text x="425" y="243" textAnchor="middle" className="sub">(uses ML/DL)</text>
        </svg>

        <div className="note">
          {data?.note ??
            "Correct nesting: AI ⊃ ML ⊃ Deep Learning ⊃ LLMs. NLP is a language domain that draws from ML/DL."}
        </div>
        <p className="muted" style={{ marginBottom: 0 }}>
          ❗ A common mistake is saying <i>"ML is part of NLP"</i>. It's the other
          way around: <b>NLP uses ML</b>, and ML is a far bigger field (it also
          powers vision, recommendations, robotics…). NLP and ML <b>overlap</b>;
          neither contains the other.
        </p>
      </div>

      {data && (
        <div className="card">
          <h3>Each layer, explained</h3>
          {data.layers.map((l) => (
            <div key={l.short} className="step">
              <span className="step-label">{l.short}</span>
              <div><b>{l.name}</b> — {l.desc}</div>
              <div className="muted">e.g. {l.examples.join(", ")}</div>
            </div>
          ))}
          <div className="step">
            <span className="step-label">{data.domain.short}</span>
            <div><b>{data.domain.name}</b> — {data.domain.desc}</div>
          </div>
          <div className="note">
            🎯 <b>Final goal — {data.goal.name}:</b> {data.goal.desc}
          </div>
        </div>
      )}
    </div>
  );
}
