import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { applyMonacoAccent } from "./monaco-setup";

// Register the monochrome + accent Monaco theme before any editor mounts.
applyMonacoAccent("#34d399");
import "./index.css";
import App from "./App";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
