import assert from "node:assert/strict";
import test from "node:test";

import { FoveaBufferError, NdjsonDecoder } from "../src/index.js";

test("frames split UTF-8 NDJSON without corrupting code points", () => {
  const decoder = new NdjsonDecoder(128);
  const encoded = Buffer.from('{"label":"café"}\n{"ok":true}\r\n');
  const split = encoded.indexOf(Buffer.from("é")) + 1;

  assert.deepEqual(decoder.push(encoded.subarray(0, split)), []);
  assert.deepEqual(decoder.push(encoded.subarray(split)), [
    '{"label":"café"}',
    '{"ok":true}',
  ]);
});

test("rejects lines beyond the configured byte cap", () => {
  const decoder = new NdjsonDecoder(4);
  assert.throws(() => decoder.push("12345"), FoveaBufferError);
});
