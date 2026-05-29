import { defineConfig } from 'vite'

export default defineConfig({
  server: {
    port: 5173,
    proxy: {
      '/chat':    'http://localhost:8000',
      '/image':   'http://localhost:8000',
      '/history': 'http://localhost:8000',
      '/health':  'http://localhost:8000',
    }
  }
})
