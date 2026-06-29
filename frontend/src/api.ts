// Tiny API client — all calls to the Python backend live here.
// One place to change if the backend URL ever moves.

const BASE_URL = "http://localhost:8000";

async function getJSON<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`);
  if (!res.ok) throw new Error(`GET ${path} failed: ${res.status}`);
  return res.json();
}

async function postJSON<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`POST ${path} failed: ${res.status}`);
  return res.json();
}

// ---- Types mirroring the backend responses (kept loose on purpose) ----------
export interface Hierarchy {
  layers: { name: string; short: string; desc: string; examples: string[] }[];
  domain: { name: string; short: string; desc: string; examples: string[] };
  goal: { name: string; short: string; desc: string };
  note: string;
}

export interface TokenRow {
  token: string;
  id: number;
  one_hot: number[];
  embedding: number[];
  hidden: {
    weights: number[][];
    bias: number[];
    pre_activation: number[];
    output: number[];
    formula: string;
  };
}

export interface PipelineResult {
  input: string;
  steps: {
    "1_normalized": string;
    "2_tokens": string[];
    "3_vocab": Record<string, number>;
    "4_to_6_tokens": TokenRow[];
    "7_cooccurrence": {
      labels: string[];
      matrix: number[][];
      total_pairs: number;
      grid_size: string;
    };
  };
  config: { embedding_dim: number; hidden_dim: number };
  explainers: { embedding: string; hidden: string; cooccurrence: string };
  field_map: { step: string; field: string; why: string }[];
  response: string;
}

export const api = {
  hierarchy: () => getJSON<Hierarchy>("/api/concepts/hierarchy"),
  pipeline: (text: string) =>
    postJSON<PipelineResult>("/api/text/pipeline", { text }),
};
