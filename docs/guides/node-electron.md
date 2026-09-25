# Node and Electron host guide

Non-Python hosts run Fovea as a child process and talk to it over the [NDJSON wire protocol](../protocol.md):

```bash
fovea run --ndjson --no-display
```

- stdout carries the `hello` handshake and then one event per line;
- stdin takes one control message per line (`calibrate`, `test`, `targets`, `observe`, `pause`, `resume`, `quit`);
- stderr is for logs;
- the exit code is `0` on normal completion, `2` for usage or configuration errors, `3` for camera errors, and `4` for
  model or runtime errors. Fatal errors are also written as a single JSON object on stdout.

With `--no-display`, the host renders every `calibration_cue` at its normalized `x`/`y` and must show a visible
camera indicator while tracking is active.

## TypeScript client

[`packages/fovea-client`](https://github.com/Wega-Labs/fovea/tree/main/packages/fovea-client) is a
zero-runtime-dependency TypeScript client. It validates the handshake, frames bounded NDJSON, maps process failures to
typed errors, coalesces gaze events, converts display coordinates, and drives host-rendered calibration. Its types are
generated from the same JSON Schema as the [protocol message reference](../reference/protocol-messages.md). The
intended package name is `@wega-labs/fovea-client`; until a maintainer publishes it, depend on the source directory.

## Electron example

[`examples/electron-host`](https://github.com/Wega-Labs/fovea/tree/main/examples/electron-host) keeps the camera
indicator visible, registers a DOM button for target-aware dwell, maps display-normalized gaze into the window, and
renders calibration cues itself:

```bash
cd examples/electron-host
npm install
FOVEA_BINARY=/absolute/path/to/fovea npm start
```

It needs the Python package and the downloaded model, plus a camera.
