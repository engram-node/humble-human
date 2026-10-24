import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    host: "127.0.0.1",
    port: 5174, // 5173 is occupied by another long-running vite on this machine
  },
  build: {
    chunkSizeWarningLimit: 5000, // monaco-editor is inherently large
    rollupOptions: {
      output: {
        manualChunks: { monaco: ["monaco-editor"] },
      },
    },
  },
});
