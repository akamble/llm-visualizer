import { NavLink, Route, Routes } from "react-router-dom";
import { BUILT, NAV } from "./modules";

export default function App() {
  return (
    <div className="app">
      <header className="topbar">
        <h1>🧠 LLM Visualizer</h1>
        <span className="tagline">Learn AI from basics to Agentic AI</span>
      </header>

      <div className="body">
        <aside className="sidebar">
          <div className="sidebar-title">Curriculum</div>
          <nav>
            {NAV.map((m) => {
              const badge = m.num === null ? "★" : m.num;
              if (m.status === "planned") {
                return (
                  <span key={m.path} className="navitem planned" title="Coming soon">
                    <span className="num-badge">{badge}</span>
                    <span className="navitem-label">{m.short}</span>
                    <span className="soon">soon</span>
                  </span>
                );
              }
              return (
                <NavLink
                  key={m.path}
                  to={m.path}
                  end={m.path === "/"}
                  className={({ isActive }) => "navitem" + (isActive ? " active" : "")}
                >
                  <span className="num-badge">{badge}</span>
                  <span className="navitem-label">{m.short}</span>
                </NavLink>
              );
            })}
          </nav>
        </aside>

        <main className="content">
          <Routes>
            {BUILT.map((m) => {
              const C = m.component!;
              return <Route key={m.path} path={m.path} element={<C />} />;
            })}
          </Routes>
        </main>
      </div>

      <footer className="footer">runs 100% locally · no API keys · no cloud</footer>
    </div>
  );
}
