import assert from "node:assert/strict";
import test from "node:test";

import { isOffWindow, toClientPoint } from "../src/index.js";

const display = { x: 1920, y: 0, width: 1920, height: 1080 };
const content = { x: 2200, y: 100, width: 900, height: 600 };

test("maps display-normalized gaze into Electron client coordinates", () => {
  const point = toClientPoint({ x: 0.25, y: 0.5 }, display, content);
  assert.deepEqual(point, { x: 200, y: 440 });
});

test("detects points outside the window content rectangle", () => {
  assert.equal(isOffWindow({ x: 0.25, y: 0.5 }, display, content), false);
  assert.equal(isOffWindow({ x: 0.95, y: 0.5 }, display, content), true);
});

test("rejects invalid host geometry", () => {
  assert.throws(
    () => toClientPoint({ x: 0.5, y: 0.5 }, display, { ...content, width: 0 }),
    RangeError,
  );
});
