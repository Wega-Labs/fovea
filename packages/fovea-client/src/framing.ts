import { FoveaBufferError } from "./errors.js";

const NEWLINE = 0x0a;
const CARRIAGE_RETURN = 0x0d;

export class NdjsonDecoder {
  private buffered: Buffer<ArrayBufferLike> = Buffer.alloc(0);

  constructor(readonly maxBytes: number) {
    if (!Number.isSafeInteger(maxBytes) || maxBytes < 1) {
      throw new RangeError("maxBytes must be a positive safe integer");
    }
  }

  push(chunk: Buffer | string): Array<string> {
    const bytes = typeof chunk === "string" ? Buffer.from(chunk) : chunk;
    const combined =
      this.buffered.length === 0 ? bytes : Buffer.concat([this.buffered, bytes]);
    const lines: Array<string> = [];
    let start = 0;
    for (let index = 0; index < combined.length; index += 1) {
      if (combined[index] !== NEWLINE) continue;
      let end = index;
      if (end > start && combined[end - 1] === CARRIAGE_RETURN) end -= 1;
      const line = combined.subarray(start, end);
      if (line.length > this.maxBytes) throw new FoveaBufferError(this.maxBytes);
      lines.push(line.toString("utf8"));
      start = index + 1;
    }
    this.buffered = combined.subarray(start);
    if (this.buffered.length > this.maxBytes) throw new FoveaBufferError(this.maxBytes);
    return lines;
  }

  finish(): Array<string> {
    if (this.buffered.length === 0) return [];
    if (this.buffered.length > this.maxBytes) throw new FoveaBufferError(this.maxBytes);
    const line = this.buffered.toString("utf8");
    this.buffered = Buffer.alloc(0);
    return [line];
  }
}
