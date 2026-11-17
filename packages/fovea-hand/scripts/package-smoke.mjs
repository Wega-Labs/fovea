// Prove the published layout works in a host outside this repository, using only its public exports,
// and that Vite serves the module worker plus its WASM/model assets in development for both a linked
// (file:) package and an installed tarball. Negative runs prove the two consumer settings are needed.
import assert from 'node:assert/strict'
import { spawnSync } from 'node:child_process'
import { cpSync, existsSync, mkdtempSync, readdirSync, readFileSync, statSync, writeFileSync } from 'node:fs'
import { createRequire } from 'node:module'
import { tmpdir } from 'node:os'
import { dirname, join, relative, resolve, sep } from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'

const PACKAGE = '@wega-labs/fovea-hand'
const root = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const example = join(root, 'examples/basic')
const scratch = mkdtempSync(join(tmpdir(), 'fovea-hand-consumer-'))
const host = join(scratch, 'host')

const run = (cmd, args, cwd = root) => {
  const result = spawnSync(cmd, args, { cwd, encoding: 'utf8', timeout: 180000 })
  if (result.status !== 0) throw Error(`${cmd} ${args.join(' ')}\n${result.stdout}\n${result.stderr}`)
  return result.stdout
}

// 1. Pack the package and install the tarball into a copy of the example, outside this repository.
run('npm', ['run', 'build'])
const modelBytes = statSync(join(root, 'dist/hand_landmarker.task')).size
const packed = JSON.parse(run('npm', ['pack', '--ignore-scripts', '--json', '--pack-destination', scratch]))[0]
cpSync(example, host, {
  recursive: true,
  filter: (src) => !['node_modules', 'dist'].includes(relative(example, src).split(sep)[0])
})
const pkg = JSON.parse(readFileSync(join(host, 'package.json'), 'utf8'))
pkg.dependencies[PACKAGE] = `file:${join(scratch, packed.filename)}`
writeFileSync(join(host, 'package.json'), JSON.stringify(pkg, null, 2) + '\n')
run('npm', ['install', '--ignore-scripts', '--no-audit', '--no-fund'], host)

// 2. The core imports in Node without DOM globals; the browser example typechecks and builds.
run(
  process.execPath,
  [
    '--input-type=module',
    '-e',
    `import assert from 'node:assert/strict';import{HandGesture,HandTargeting,HoldToPlace}from'${PACKAGE}';assert.equal(typeof document,'undefined');assert.equal(new HandGesture(1).phase,'idle');assert.equal(new HandTargeting().dwellMs,700);assert.ok(new HoldToPlace({x:0,y:0},0));`
  ],
  host
)
run(
  'npm',
  [
    'exec',
    '--',
    'tsc',
    '--noEmit',
    '--strict',
    '--skipLibCheck',
    '--module',
    'ESNext',
    '--moduleResolution',
    'bundler',
    '--target',
    'ES2022',
    '--lib',
    'ES2023,DOM,DOM.Iterable',
    'main.ts'
  ],
  host
)
run('npm', ['run', 'build'], host)
const assets = readdirSync(join(host, 'dist/assets'))
assert.ok(assets.some((name) => /^tracker\.worker-.*\.js$/.test(name)), 'worker bundled')
assert.ok(assets.some((name) => name.endsWith('.wasm')), 'WASM bundled')
assert.ok(assets.some((name) => name.endsWith('.task')), 'model bundled')
const installed = join(host, 'node_modules', PACKAGE)
for (const file of [
  'LICENSE',
  'dist/MODEL_LICENSE.md',
  'dist/LICENSES/mediapipe-Apache-2.0.txt',
  'dist/LICENSES/one-euro-BSD-3-Clause.txt'
])
  assert.ok(existsSync(join(installed, file)), `${file} ships in the package`)
assert.ok(!existsSync(join(host, 'node_modules/react')), 'host has no React')
assert.ok(!existsSync(join(host, 'node_modules/electron')), 'host has no Electron')

// 3. Development serving. Vite whitelists only files reached through static import analysis; the
//    module worker and the `?url` assets are resolved separately and depend on `server.fs.allow`
//    (linked layout) and `optimizeDeps.exclude` (installed layout). Fetch over HTTP like a browser.
//    Vite treats `server.port: 0` as unset, so it picks its own free port; read it from the socket.
const WORKER_URL = /new\s+Worker\(\s*new\s+URL\(\s*(?:\/\*\s*@vite-ignore\s*\*\/\s*)?["']([^"']+)["']\s*,\s*import\.meta\.url\s*\)/
const importOf = (body, pattern) => body.match(new RegExp(`from\\s+["']([^"']*${pattern}[^"']*)["']`))?.[1]

/** Returns { worker: 'served' } after checking every asset, or { worker: 'rejected', status } when
 *  the worker request itself fails. Anything else (a broken module walk, a bad asset) throws. */
async function checkDevServing(root, config) {
  const require = createRequire(join(root, 'package.json'))
  const { createServer } = await import(pathToFileURL(require.resolve('vite')).href)
  const server = await createServer({
    root,
    configFile: config.configFile,
    logLevel: 'silent',
    // Fetch-only: background pre-transforms of optimized deps would keep close() from settling.
    server: { host: '127.0.0.1', strictPort: false, preTransformRequests: false },
    ...config.inline
  })
  try {
    await server.listen()
    const origin = `http://127.0.0.1:${server.httpServer.address().port}`
    const get = async (path) => {
      const response = await fetch(new URL(path, origin))
      const body = Buffer.from(await response.arrayBuffer())
      const type = response.headers.get('content-type') ?? ''
      return { status: response.status, body, html: /text\/html/.test(type) || /^\s*<!doctype html/i.test(body.toString('utf8', 0, 64)) }
    }
    const module = async (path) => {
      const response = await get(path)
      assert.equal(response.status, 200, `${path} served`)
      assert.ok(!response.html, `${path} is a module, not the HTML fallback`)
      return { path, body: response.body.toString('utf8') }
    }
    const main = await module('/main.ts')
    const browserPath = importOf(main.body, 'fovea-hand[^"\']*browser')
    assert.ok(browserPath, 'main.ts imports the browser entry')
    let entry = await module(browserPath)
    if (!WORKER_URL.test(entry.body)) {
      const cameraPath = importOf(entry.body, 'camera\\.js')
      assert.ok(cameraPath, 'browser entry re-exports camera.js')
      entry = await module(cameraPath)
    }
    const workerRef = entry.body.match(WORKER_URL)?.[1]
    assert.ok(workerRef, 'camera module constructs the worker from a URL')
    const workerPath = new URL(workerRef, new URL(entry.path, origin)).pathname + new URL(workerRef, origin).search
    const worker = await get(workerPath)
    if (worker.status !== 200 || worker.html) return { worker: 'rejected', status: worker.status, html: worker.html }
    const workerBody = worker.body.toString('utf8')
    assert.ok(workerBody.includes('HandLandmarker'), 'worker body is the tracker module')
    const wrappers = [...workerBody.matchAll(/import\s+\w+\s+from\s+["']([^"']*[?&]url\b[^"']*)["']/g)].map((m) => m[1])
    assert.equal(wrappers.length, 3, `worker imports three ?url assets: ${wrappers.join(', ')}`)
    const checked = []
    for (const wrapperPath of wrappers) {
      const wrapper = await module(wrapperPath)
      const assetPath = wrapper.body.match(/export default\s+"([^"]+)"/)?.[1]
      assert.ok(assetPath, `${wrapperPath} exports its URL`)
      const asset = await get(assetPath)
      assert.equal(asset.status, 200, `${assetPath} served`)
      assert.ok(!asset.html, `${assetPath} is not the HTML fallback`)
      const file = assetPath.split('?')[0]
      if (file.endsWith('.wasm')) assert.equal(asset.body.subarray(0, 4).toString('latin1'), '\0asm', 'WASM magic')
      else if (file.endsWith('.task')) assert.equal(asset.body.length, modelBytes, 'model byte length')
      else assert.ok(file.endsWith('.js'), `unexpected asset ${file}`)
      checked.push(file)
    }
    assert.ok(checked.some((f) => f.endsWith('.wasm')) && checked.some((f) => f.endsWith('.task')) && checked.some((f) => f.endsWith('.js')))
    // Every prebundled dependency must come from the shipped `optimizeDeps.include` entries, whose
    // keys keep the `package > dependency` form. A plain `1eurofilter` or `@mediapipe/tasks-vision`
    // key means Vite discovered it while serving, which in a browser forces a re-optimization and
    // a full page reload on first use. Vite moves entries between these two maps as it commits.
    const { optimized, discovered } = server.environments.client.depsOptimizer.metadata
    const prebundled = [...Object.keys(optimized), ...Object.keys(discovered)].sort()
    assert.deepEqual(prebundled, [`${PACKAGE} > 1eurofilter`, `${PACKAGE} > @mediapipe/tasks-vision`], 'no runtime dependency discovery')
    return { worker: 'served', assets: checked, prebundled }
  } finally {
    await server.close()
  }
}

// Without the shipped config, keep only the worker format so the failure isolates the tested setting.
// `force` discards the previous run's prebundle cache so it cannot decide the outcome.
const bare = { configFile: false, inline: { worker: { format: 'es' }, optimizeDeps: { force: true } } }
const linked = await checkDevServing(example, { configFile: join(example, 'vite.config.ts') })
assert.equal(linked.worker, 'served', 'linked package: shipped config serves worker and assets')
const linkedBare = await checkDevServing(example, bare)
assert.equal(linkedBare.worker, 'rejected', 'linked package without fs.allow: worker request must be rejected')
assert.equal(linkedBare.status, 403, 'linked package without fs.allow: /@fs/ access is forbidden')
const tarball = await checkDevServing(host, { configFile: join(host, 'vite.config.ts') })
assert.equal(tarball.worker, 'served', 'installed package: shipped config serves worker and assets')
const tarballBare = await checkDevServing(host, bare)
assert.equal(tarballBare.worker, 'rejected', 'installed package without optimizeDeps.exclude: worker request must fail')
assert.equal(tarballBare.status, 404, 'installed package without optimizeDeps.exclude: prebundled worker URL is missing')

const evidence = {
  tarball: packed.filename,
  packageBytes: packed.size,
  checks: [
    'core imports in Node without DOM globals',
    'standalone browser host typechecks and builds using public package exports',
    'worker, WASM, model, LICENSE and third-party licenses included',
    'host has no React or Electron dependency',
    'dev server serves worker and assets for linked and installed layouts with the shipped config',
    'dev server rejects the worker without fs.allow (403, linked) and without optimizeDeps.exclude (404, installed)'
  ],
  devServing: { linked, linkedBare, tarball, tarballBare },
  host
}
writeFileSync(join(scratch, 'evidence.json'), JSON.stringify(evidence, null, 2) + '\n')
console.log(JSON.stringify(evidence, null, 2))
console.log(`Evidence: ${join(scratch, 'evidence.json')}`)
