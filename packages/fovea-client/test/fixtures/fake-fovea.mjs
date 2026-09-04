import { existsSync, writeFileSync } from "node:fs";
import { createInterface } from "node:readline";

const arguments_ = process.argv.slice(2);
const option = (name, fallback = "") =>
  arguments_.find((value) => value.startsWith(`--${name}=`))?.slice(name.length + 3) ?? fallback;
const scenario = option("scenario", "normal");
const hello = JSON.stringify({
  type: "hello",
  protocol: "1.1",
  fovea: "test",
  backend: "fake",
  coordinate_space: "display_normalized",
  indicator_required: true,
  capabilities: ["calibration_cue", "fixation"],
});

function send(value) {
  process.stdout.write(`${JSON.stringify(value)}\n`);
}

function start() {
  if (scenario === "wrong-protocol") {
    send({ ...JSON.parse(hello), protocol: "2.0" });
    return;
  }
  if (scenario === "oversized") {
    process.stdout.write("x".repeat(512));
    return;
  }
  if (scenario === "split") {
    const midpoint = Math.floor(hello.length / 2);
    process.stdout.write(hello.slice(0, midpoint));
    setTimeout(() => process.stdout.write(`${hello.slice(midpoint)}\n`), 5);
  } else {
    process.stdout.write(`${hello}\n`);
  }

  if (scenario === "malformed") {
    setTimeout(() => process.stdout.write("{not-json}\n"), 15);
  } else if (scenario === "events") {
    setTimeout(() => {
      send({
        type: "tracking_state",
        status: "active",
        confidence: 0.9,
        timestamp_ns: 1,
      });
      send({
        type: "fixation",
        x: 0.5,
        y: 0.5,
        duration_ms: 300,
        confidence: 0.8,
        timestamp_ns: 2,
      });
    }, 15);
  } else if (scenario === "burst") {
    setTimeout(() => {
      for (const x of [0.1, 0.2, 0.3]) {
        send({ type: "gaze_point", x, y: 0.5, confidence: 0.9, timestamp_ns: x * 10 });
      }
    }, 15);
  } else if (scenario === "crash") {
    setTimeout(() => process.exit(4), 20);
  } else if (scenario === "reconnect-once") {
    const marker = option("marker");
    if (!existsSync(marker)) {
      writeFileSync(marker, "first attempt\n");
      setTimeout(() => process.exit(4), 20);
    }
  }
}

const input = createInterface({ input: process.stdin });
input.on("line", (line) => {
  const command = JSON.parse(line);
  if (command.cmd === "calibrate") {
    send({
      type: "calibration_cue",
      label: "center",
      x: 0.5,
      y: 0.5,
      index: 0,
      total: 5,
      samples: 1,
      needed: 1,
      instruction: "Look at the point.",
      timestamp_ns: 3,
    });
    setTimeout(() => {
      send({
        type: "calibration_done",
        n_points: 5,
        coverage: 0.64,
        loo_error: 0.05,
        timestamp_ns: 4,
      });
    }, 5);
  } else if (command.cmd === "observe") {
    const marker = option("marker");
    if (marker !== "") writeFileSync(marker, JSON.stringify(command));
    send({ type: "calibration_updated", n: 5, loo_error: 0.04, timestamp_ns: 5 });
  } else if (command.cmd === "quit") {
    process.exit(0);
  }
});

start();
