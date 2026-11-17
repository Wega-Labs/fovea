import { realpathSync } from 'node:fs'
import { resolve } from 'node:path'
import { defineConfig, searchForWorkspaceRoot } from 'vite'

export default defineConfig({
  base: './',
  // MediaPipe's WASM loader is an ES module; the default IIFE worker cannot import it.
  worker: { format: 'es' },
  // Prebundling would inline the browser entry and break its relative module-worker URL. Excluding
  // the package also stops Vite's scanner at its boundary, so list its own dependencies (1eurofilter
  // is CommonJS and must be prebundled) or their first use triggers a re-optimization and a reload.
  optimizeDeps: {
    exclude: ['@wega-labs/fovea-hand'],
    include: ['@wega-labs/fovea-hand > @mediapipe/tasks-vision', '@wega-labs/fovea-hand > 1eurofilter']
  },
  // A linked (file:) package lives outside the Vite root; its worker, WASM and model are served
  // through /@fs/ and must be allowed explicitly. Resolve the real path, never a relative guess.
  server: {
    fs: {
      allow: [searchForWorkspaceRoot(__dirname), realpathSync(resolve(__dirname, 'node_modules/@wega-labs/fovea-hand'))]
    }
  }
})
