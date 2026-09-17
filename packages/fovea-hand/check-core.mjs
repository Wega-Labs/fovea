import assert from 'node:assert/strict'
import { dirname, isAbsolute, relative, resolve, sep } from 'node:path'
import { fileURLToPath } from 'node:url'
import ts from 'typescript'

const root = dirname(fileURLToPath(import.meta.url))
const configPath = resolve(root, 'tsconfig.core.json')
const config = ts.readConfigFile(configPath, ts.sys.readFile)
const parsed = ts.parseJsonConfigFileContent(config.config, ts.sys, root)
const program = ts.createProgram(parsed.fileNames, parsed.options)
const diagnostics = [config.error, ...parsed.errors, ...ts.getPreEmitDiagnostics(program)].filter(Boolean)
if (diagnostics.length) {
  console.error(
    ts.formatDiagnosticsWithColorAndContext(diagnostics, {
      getCurrentDirectory: () => root,
      getCanonicalFileName: (name) => name,
      getNewLine: () => '\n'
    })
  )
  process.exit(1)
}

// Follow TypeScript's complete source/type graph, including re-exports and type-only
// imports. A successful no-DOM compile alone would still permit unrelated pure app code.
const browserFiles = new Set(['browser.ts', 'camera.ts', 'dom-targets.ts', 'skeleton.ts', 'tracker.worker.ts'])
let coreFiles = 0
for (const source of program.getSourceFiles()) {
  if (program.isSourceFileDefaultLibrary(source)) {
    assert.match(
      source.fileName,
      /[/\\]lib\.(?:es[^/\\]*|decorators(?:\.legacy)?)\.d\.ts$/,
      'core must use only ECMAScript ambient libraries'
    )
    continue
  }
  const name = relative(resolve(root, 'src'), source.fileName)
  if (name !== '..' && !name.startsWith(`..${sep}`) && !isAbsolute(name)) {
    assert.ok(!browserFiles.has(name) && !name.endsWith('.test.ts'), `core reaches browser/test module: ${name}`)
    coreFiles++
    // Nonliteral imports cannot be resolved and audited by the compiler graph.
    const visit = (node) => {
      if (ts.isCallExpression(node) && node.expression.kind === ts.SyntaxKind.ImportKeyword)
        assert.ok(
          node.arguments.length === 1 && ts.isStringLiteral(node.arguments[0]),
          `nonliteral core import in ${name}`
        )
      ts.forEachChild(node, visit)
    }
    visit(source)
    continue
  }
  // This is the core's sole intentional external dependency. Browser inference
  // dependencies, React/Electron, Node types and October source are forbidden.
  assert.match(
    source.fileName,
    /[/\\]node_modules[/\\]1eurofilter[/\\]/,
    `core dependency escaped package: ${source.fileName}`
  )
}
console.log(`Hand core boundary passed: ${coreFiles} source files, ES2023 only, no DOM/React/Electron/October imports`)
