// Central registry of all modules — the single source of truth for navigation,
// routing, and the roadmap index. Adding a module = add ONE entry here (plus its
// page component). The sidebar and router both render from this list, so the
// growing curriculum stays properly indexed and in order.

import type { ComponentType } from "react";
import BigPicture from "./pages/BigPicture";
import TextPipeline from "./pages/TextPipeline";
import Tokenization from "./pages/Tokenization";
import Embeddings from "./pages/Embeddings";
import Positional from "./pages/Positional";

export interface ModuleDef {
  num: number | null; // null = not a numbered module (the Big Picture overview)
  path: string;
  short: string; // sidebar label
  title: string; // full page title
  status: "done" | "planned";
  component?: ComponentType; // present only for built modules
}

export const NAV: ModuleDef[] = [
  { num: null, path: "/", short: "Big Picture", title: "The Big Picture", status: "done", component: BigPicture },
  { num: 1, path: "/pipeline", short: "Text → Response", title: "How text becomes a response", status: "done", component: TextPipeline },
  { num: 2, path: "/tokenization", short: "Tokenization (BPE)", title: "Tokenization with BPE", status: "done", component: Tokenization },
  { num: 3, path: "/embeddings", short: "Embeddings & Similarity", title: "Embeddings & Similarity", status: "done", component: Embeddings },
  { num: 4, path: "/positional", short: "Positional Encoding", title: "Positional Encoding", status: "done", component: Positional },

  // Planned — shown in the index so the roadmap is visible, but not yet clickable.
  { num: 5, path: "/attention", short: "Self-Attention (Q/K/V)", title: "Self-Attention", status: "planned" },
  { num: 6, path: "/softmax", short: "Logits & Softmax", title: "Logits & Softmax", status: "planned" },
  { num: 7, path: "/generation", short: "Sampling & Generation", title: "Sampling & Generation", status: "planned" },
  { num: 8, path: "/gpt2", short: "Real Model (GPT-2)", title: "Using a Real Model", status: "planned" },
  { num: 9, path: "/rag", short: "RAG", title: "Retrieval-Augmented Generation", status: "planned" },
  { num: 10, path: "/agents", short: "Agentic AI", title: "Agentic AI", status: "planned" },
];

export const BUILT = NAV.filter((m) => m.component);
