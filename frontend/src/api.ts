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

export interface MergeStep {
  step: number;
  pair: [string, string];
  merged: string;
  count: number;
  vocab_size: number;
  example: { word: string; before: string[]; after: string[] };
}

export interface TokenizationResult {
  input: string;
  num_merges: number;
  corpus: Record<string, number>;
  training: {
    merges: [string, string][];
    history: MergeStep[];
    base_vocab: string[];
    final_vocab: string[];
  };
  encoding: {
    per_word: { word: string; tokens: string[] }[];
    tokens: string[];
  };
  comparison: {
    word_level_vocab: number;
    char_level_vocab: number;
    bpe_vocab: number;
    input_word_tokens: number;
    input_bpe_tokens: number;
  };
  explainers: {
    why_subword: string;
    training: string;
    encoding: string;
    end_of_word: string;
  };
  field_map: { step: string; field: string; why: string }[];
  code_sample: string;
  response: string;
}

export interface EmbeddingsResult {
  documents: string[];
  query: string;
  vocab: string[];
  tf: number[][];
  idf: Record<string, number>;
  tfidf: number[][];
  query_vector: number[];
  similarity_matrix: number[][];
  ranking: { doc_index: number; doc: string; score: number }[];
  worked_example: {
    a_index: number;
    b_index: number;
    a_doc: string;
    b_doc: string;
    dot: number;
    norm_a: number;
    norm_b: number;
    score: number;
    formula: string;
  };
  explainers: { tfidf: string; cosine: string; search: string };
  field_map: { step: string; field: string; why: string }[];
  code_sample: string;
  response: string;
}

export interface PositionalResult {
  input: string;
  dim: number;
  tokens: string[];
  pe_matrix: number[][];
  per_token: {
    position: number;
    token: string;
    embedding: number[];
    positional: number[];
    result: number[];
  }[];
  position_similarity: number[][];
  worked_example: {
    pos: number;
    even: { dim: number; formula: string; calc: string };
    odd: { dim: number; formula: string; calc: string };
    note: string;
  };
  order_demo: {
    a: { sentence: string; vectors: { token: string; position: number; result: number[] }[] };
    b: { sentence: string; vectors: { token: string; position: number; result: number[] }[] };
    note: string;
  };
  explainers: { why: string; how: string; added: string };
  field_map: { step: string; field: string; why: string }[];
  code_sample: string;
  response: string;
}

export const api = {
  hierarchy: () => getJSON<Hierarchy>("/api/concepts/hierarchy"),
  pipeline: (text: string) =>
    postJSON<PipelineResult>("/api/text/pipeline", { text }),
  tokenize: (text: string, num_merges: number) =>
    postJSON<TokenizationResult>("/api/text/tokenize", { text, num_merges }),
  embeddings: (documents: string[], query: string) =>
    postJSON<EmbeddingsResult>("/api/text/embeddings", { documents, query }),
  positional: (text: string, dim: number) =>
    postJSON<PositionalResult>("/api/text/positional", { text, dim }),
};
