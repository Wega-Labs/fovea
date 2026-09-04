# `@wega-labs/fovea-client`

Typed Node.js and Electron process client for Fovea's versioned NDJSON protocol.
It has zero runtime dependencies. The npm package is not published yet; publishing
under the Wega Labs organization is a maintainer action.

## What it owns

- spawning `fovea run --ndjson` directly or through `python -m fovea`
- validating the required `hello` handshake and protocol major
- bounded, split-safe UTF-8 NDJSON framing
- typed EventEmitter and AsyncIterable event delivery
- stable process-exit errors and optional exponential-backoff reconnect
- latest-value gaze coalescing at a configurable maximum rate
- display-normalized to Electron client-coordinate mapping
- calibration cues and completion as one Promise-based workflow

Camera frames stay inside the Fovea process. The client receives only protocol
events and does not add analytics, telemetry, or network access.

## Build from this repository

```bash
cd packages/fovea-client
npm install
npm run generate:check
npm run check
npm test
npm run build
```

The generated event and command interfaces in `src/generated/protocol.ts` come
from `schema/fovea-protocol-v1.json`. Run `npm run generate` after an additive
protocol change and commit both files together.

## Spawn and consume events

```ts
import { spawnFovea } from "@wega-labs/fovea-client";

const fovea = spawnFovea({
  binary: "/path/to/fovea",
  camera: 0,
  args: ["--no-display"],
  maxGazeHz: 30,
});

const hello = await fovea.ready;
console.log(`Fovea ${hello.fovea}, backend ${hello.backend}`);

fovea.on("fixation", (event) => {
  console.log(event.x, event.y, event.duration_ms);
});

for await (const event of fovea) {
  if (event.type === "dwell") console.log(`activate ${event.id}`);
}
```

To use a Python environment instead of an installed console script:

```ts
const fovea = spawnFovea({ python: "/project/.venv/bin/python" });
```

For launchers, `prefixArgs` are placed before `run --ndjson`:

```ts
const fovea = spawnFovea({ binary: "uv", prefixArgs: ["run", "fovea"] });
```

`close()` sends `quit` and then terminates the full child process tree. Call it
from every application shutdown path.

## Reconnect

The client's top-level `reconnect` option restarts the entire Fovea child process after it exits.
Fovea's `--reconnect` CLI flag instead keeps one process alive while it waits for a successfully
opened camera to return. Pass camera name/id selection and camera-level reconnect through `args`:

```ts
const fovea = spawnFovea({
  args: ["--camera-id", "stable-id-from-doctor", "--reconnect"],
});
```

Use the process-level option when the host should recover from any child-process camera exit:

```ts
const fovea = spawnFovea({
  reconnect: {
    initialDelayMs: 250,
    maxDelayMs: 4_000,
    factor: 2,
    maxAttempts: 5,
  },
});
```

Reconnect never silently replays calibration or registered target rectangles.
Listen for the next `hello` and explicitly restore host state once the new
process is ready.

## Coordinates

Electron's `screen.getDisplayMatching(window.getBounds()).bounds` supplies the
display rectangle, while `window.getContentBounds()` supplies the content
rectangle. Both must use the same desktop coordinate system.

```ts
import { isOffWindow, toClientPoint } from "@wega-labs/fovea-client";

fovea.on("gaze_point", (event) => {
  if (isOffWindow(event, displayBounds, contentBounds)) return;
  const point = toClientPoint(event, displayBounds, contentBounds);
  window.webContents.send("fovea:gaze", point);
});
```

## Calibration

The host remains responsible for rendering the cue and a persistent camera-use
indicator.

```ts
const result = await fovea.runCalibration({
  targets,
  onCue: (cue) => window.webContents.send("fovea:calibration-cue", cue),
  onWarning: (warning) => console.warn(warning.message),
});
```

See `examples/electron-host` for a minimal target registration, dwell, gaze
overlay, and window-rendered calibration flow.
