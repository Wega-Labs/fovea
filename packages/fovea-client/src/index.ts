export { FoveaClient, spawnFovea } from "./client.js";
export {
  FoveaBufferError,
  FoveaClientError,
  FoveaClosedError,
  FoveaHandshakeError,
  FoveaProcessError,
  exitReason,
} from "./errors.js";
export { NdjsonDecoder } from "./framing.js";
export { isOffWindow, toClientPoint } from "./coordinates.js";
export * from "./types.js";
