import { Link, Route, Routes } from "react-router-dom";
import BigPicture from "./pages/BigPicture";
import TextPipeline from "./pages/TextPipeline";

export default function App() {
  return (
    <div className="app">
      <header className="topbar">
        <h1>🧠 LLM Visualizer</h1>
        <nav>
          <Link to="/">Big Picture</Link>
          <Link to="/pipeline">Lesson 1: Text → Response</Link>
        </nav>
      </header>

      <main className="content">
        <Routes>
          <Route path="/" element={<BigPicture />} />
          <Route path="/pipeline" element={<TextPipeline />} />
        </Routes>
      </main>

      <footer className="footer">
        Learn AI from basics to Agentic AI · runs 100% locally
      </footer>
    </div>
  );
}
