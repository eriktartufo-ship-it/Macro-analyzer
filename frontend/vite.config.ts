import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

declare const process: { env: Record<string, string | undefined> };

// MACRO_API_TARGET permette di guardare il frontend locale sui dati veri
// (es. https://macro.pieranlab.cloud): le scritture restano protette dal token admin.
const apiTarget = process.env.MACRO_API_TARGET ?? "http://localhost:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      "/api": {
        target: apiTarget,
        changeOrigin: true,
        secure: true,
      },
    },
  },
});
