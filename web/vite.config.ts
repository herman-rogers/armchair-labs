import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    // The FastAPI read API. Proxying keeps the frontend origin-relative, so the same
    // build works behind a reverse proxy in the homelab without a rebuild.
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
})
