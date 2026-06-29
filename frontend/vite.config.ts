import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Vite dev server runs on http://localhost:5173 and proxies nothing —
// the frontend calls the FastAPI backend directly at http://localhost:8000.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
  },
});
