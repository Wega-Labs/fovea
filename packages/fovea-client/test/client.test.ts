import assert from "node:assert/strict";
import { once } from "node:events";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";

import {
  FoveaBufferError,
  FoveaHandshakeError,
  FoveaProcessError,
  spawnFovea,
} from "../src/index.js";
import type { FoveaClient, SpawnFoveaOptions } from "../src/index.js";

const fixture = fileURLToPath(new URL("./fixtures/fake-fovea.mjs", import.meta.url));

function spawnFake(
  scenario: string,
  options: Omit<SpawnFoveaOptions, "binary" | "python" | "prefixArgs"> = {},
): FoveaClient {
  return spawnFovea({
    ...options,
    binary: process.execPath,
    prefixArgs: [fixture],
    args: [`--scenario=${scenario}`, ...(options.args ?? [])],
    startTimeoutMs: options.startTimeoutMs ?? 1_000,
  });
}

test("validates a split hello and drives calibration", async (context) => {
  const client = spawnFake("split");
  context.after(() => client.close());

  const hello = await client.ready;
  assert.equal(hello.protocol, "1.0");
  assert.equal(hello.backend, "fake");
  const cues: Array<string> = [];
  const result = await client.runCalibration({
    onCue: (cue) => {
      cues.push(cue.label);
    },
  });

  assert.deepEqual(cues, ["center"]);
  assert.equal(result.n_points, 5);
  assert.equal(result.loo_error, 0.05);
});

test("is both an EventEmitter and an AsyncIterable", async (context) => {
  const client = spawnFake("events");
  context.after(() => client.close());
  const iterator = client[Symbol.asyncIterator]();
  const emitted = once(client, "tracking_state");

  await client.ready;
  const [tracking] = await emitted;
  const iterated = await iterator.next();

  assert.equal(tracking.type, "tracking_state");
  assert.equal(iterated.done, false);
  assert.equal(iterated.value?.type, "tracking_state");
  await iterator.return?.();
});

test("validates and writes online observations", async (context) => {
  const directory = mkdtempSync(join(tmpdir(), "fovea-client-observe-"));
  const marker = join(directory, "command.json");
  const client = spawnFake("observe", { args: [`--marker=${marker}`] });
  context.after(() => {
    client.close();
    rmSync(directory, { recursive: true, force: true });
  });
  const updated = once(client, "calibration_updated");
  await client.ready;
  assert.throws(() => client.observe(-0.1, 0.5), RangeError);
  assert.throws(() => client.observe(0.5, 0.5, 0), RangeError);
  client.observe(0.25, 0.75, 0.4, 123);
  const [event] = await updated;

  assert.equal(event.n, 5);
  assert.deepEqual(JSON.parse(readFileSync(marker, "utf8")), {
    cmd: "observe",
    x: 0.25,
    y: 0.75,
    weight: 0.4,
    timestamp_ns: 123,
  });
});

test("reports malformed event lines without losing the process", async (context) => {
  const client = spawnFake("malformed");
  context.after(() => client.close());
  const malformed = once(client, "malformedLine");

  await client.ready;
  const [notice] = await malformed;

  assert.match(notice.line, /not-json/);
  assert(notice.error instanceof SyntaxError);
  assert.equal(client.status, "running");
});

test("maps process crashes to stable exit information", async (context) => {
  const client = spawnFake("crash");
  context.after(() => client.close());
  const processError = once(client, "processError");

  await client.ready;
  const [error] = await processError;

  assert(error instanceof FoveaProcessError);
  assert.equal(error.exitCode, 4);
  assert.match(error.message, /model or runtime/);
});

test("can reconnect with bounded exponential backoff", async (context) => {
  const directory = mkdtempSync(join(tmpdir(), "fovea-client-test-"));
  const marker = join(directory, "connected");
  const client = spawnFake("reconnect-once", {
    args: [`--marker=${marker}`],
    reconnect: { initialDelayMs: 5, maxDelayMs: 10, factor: 2, maxAttempts: 2 },
  });
  context.after(() => {
    client.close();
    rmSync(directory, { recursive: true, force: true });
  });
  const attempts: Array<number> = [];
  client.on("reconnecting", (notice) => attempts.push(notice.attempt));
  let helloCount = 0;
  const reconnected = new Promise<void>((resolve) => {
    client.on("hello", () => {
      helloCount += 1;
      if (helloCount === 2) resolve();
    });
  });

  await client.ready;
  await reconnected;

  assert.deepEqual(attempts, [1]);
  assert.equal(client.status, "running");
});

test("accepts a newer protocol minor within the supported major", async (context) => {
  const client = spawnFake("normal", { args: ["--protocol=1.2"] });
  context.after(() => client.close());

  const hello = await client.ready;

  assert.equal(hello.protocol, "1.2");
  assert.equal(client.status, "running");
});

test("rejects incompatible protocol majors", async (context) => {
  const client = spawnFake("wrong-protocol");
  context.after(() => client.close());
  await assert.rejects(client.ready, FoveaHandshakeError);
});

test("caps an unterminated stdout line", async (context) => {
  const client = spawnFake("oversized", { maxBufferBytes: 64 });
  context.after(() => client.close());
  await assert.rejects(client.ready, FoveaBufferError);
});

test("coalesces gaze bursts to the latest pending point", async (context) => {
  const client = spawnFake("burst", { maxGazeHz: 10 });
  context.after(() => client.close());
  const points: Array<number> = [];
  client.on("gaze_point", (event) => points.push(event.x));

  await client.ready;
  await new Promise((resolve) => setTimeout(resolve, 150));

  assert.deepEqual(points, [0.1, 0.3]);
});
