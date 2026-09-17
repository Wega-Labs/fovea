import assert from 'node:assert/strict'
import { test } from 'node:test'
import { execFileSync, spawnSync } from 'node:child_process'
import { mkdtempSync, mkdirSync, readFileSync, readdirSync, copyFileSync, symlinkSync, writeFileSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..')

test('core boundary follows re-exports/types and rejects host code or injected DOM libraries', () => {
  const scratch = mkdtempSync(join(tmpdir(), 'fovea-hand-core-boundary-'))
  try {
    mkdirSync(join(scratch, 'src'))
    symlinkSync(join(root, 'node_modules'), join(scratch, 'node_modules'), 'junction')
    for (const name of ['check-core.mjs', 'tsconfig.json', 'tsconfig.core.json'])
      copyFileSync(join(root, name), join(scratch, name))
    for (const name of readdirSync(join(root, 'src')).filter(
      (name) => name.endsWith('.ts') && !name.endsWith('.test.ts')
    ))
      copyFileSync(join(root, 'src', name), join(scratch, 'src', name))
    const run = () =>
      spawnSync(process.execPath, ['check-core.mjs'], { cwd: scratch, encoding: 'utf8', timeout: 20000 })
    assert.equal(run().status, 0)
    const index = readFileSync(join(scratch, 'src/index.ts'), 'utf8')
    writeFileSync(join(scratch, 'host.ts'), 'export interface HostState { value: number }\n')
    for (const injection of [
      'export type { HostState } from "../host.js"',
      'export * from "./browser.js"',
      'export type { ReactNode } from "react"',
      'export type { BrowserWindow } from "electron"',
      'export interface LeakedElement { element: HTMLElement }',
      '/// <reference lib="dom" />\nexport interface InjectedDOM { element: HTMLElement }',
      'export const load = (path: string) => import(path)'
    ]) {
      // A nested re-export ensures the check follows dependencies beyond index.ts.
      writeFileSync(join(scratch, 'src/index.ts'), index + '\nexport * from "./boundary-probe.js"\n')
      writeFileSync(join(scratch, 'src/boundary-probe.ts'), injection + '\n')
      const result = run()
      assert.equal(result.status, 1, `${injection}\n${result.stdout}\n${result.stderr}`)
    }
    // Public optional modes remain usable by a standalone no-DOM consumer.
    execFileSync(
      process.execPath,
      [
        '--input-type=module',
        '-e',
        `
      import assert from 'node:assert/strict';
      import { HandGesture, HandTargeting, HandPanGesture, HandSnapGesture } from './dist/index.js';
      assert.equal(typeof document, 'undefined');
      for (const mode of ['pinch','curl']) assert.equal(new HandGesture(1,undefined,mode).phase,'idle');
      assert.ok(new HandTargeting());assert.ok(new HandPanGesture(1));assert.ok(new HandSnapGesture(1));
    `
      ],
      { cwd: root, stdio: 'pipe' }
    )
  } finally {
    rmSync(scratch, { recursive: true, force: true })
  }
})
