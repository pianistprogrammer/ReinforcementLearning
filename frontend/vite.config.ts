import { defineConfig } from 'vite';
export default defineConfig({
  build: { outDir: '../rl_framework/web/static', emptyOutDir: true },
  server: { proxy: { '/api': 'http://127.0.0.1:8765', '/socket.io': { target: 'http://127.0.0.1:8765', ws: true } } },
});
