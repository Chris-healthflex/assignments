import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The build lands in app/static, which FastAPI mounts at /ui. Serving the UI
// from the API's own origin is what lets the app call the endpoints with no
// CORS configuration anywhere: the browser never sees a cross-origin request.
//
// `base` has to match the mount point, or the built asset URLs come out
// absolute from "/" and 404 behind the /ui prefix.
export default defineConfig({
  plugins: [react()],
  base: "/ui/",
  build: {
    outDir: "../app/static",
    emptyOutDir: true,
    // One vendor chunk. The app is small enough that further splitting only
    // adds round trips.
    rollupOptions: {
      output: {
        manualChunks: { react: ["react", "react-dom"] },
      },
    },
  },
  server: {
    port: 5173,
    // `npm run dev` gives hot reload while still talking to the real API.
    proxy: {
      "/assessments": "http://127.0.0.1:8000",
      "/health": "http://127.0.0.1:8000",
    },
  },
});
