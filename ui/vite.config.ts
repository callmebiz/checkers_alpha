// Vite: the dev server (`npm run dev`) forwards /api, including the live
// analysis WebSocket, to the Python server on port 8020.
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5203, // see ../../.claude (the cross-project port registry)
    strictPort: true,
    proxy: {
      '/api': { target: 'http://127.0.0.1:8020', ws: true },
    },
  },
})
