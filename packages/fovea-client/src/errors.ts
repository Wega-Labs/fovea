const EXIT_REASONS: Readonly<Record<number, string>> = {
  0: "the Fovea process ended",
  2: "Fovea rejected its command-line or control configuration",
  3: "Fovea could not open the requested camera",
  4: "Fovea could not initialize its model or runtime",
};

export class FoveaClientError extends Error {
  constructor(
    message: string,
    readonly kind: string,
    options?: ErrorOptions,
  ) {
    super(message, options);
    this.name = new.target.name;
  }
}

export class FoveaHandshakeError extends FoveaClientError {
  constructor(message: string, options?: ErrorOptions) {
    super(message, "handshake", options);
  }
}

export class FoveaBufferError extends FoveaClientError {
  constructor(readonly maxBytes: number) {
    super(`Fovea emitted an NDJSON line larger than ${maxBytes} bytes`, "buffer_limit");
  }
}

export class FoveaClosedError extends FoveaClientError {
  constructor() {
    super("Fovea client is closed", "closed");
  }
}

export class FoveaProcessError extends FoveaClientError {
  constructor(
    readonly exitCode: number | null,
    readonly signal: NodeJS.Signals | null,
    readonly stderr: string,
    options?: ErrorOptions,
  ) {
    const reason = exitCode === null ? "the Fovea process was terminated" : exitReason(exitCode);
    const signalDetail = signal === null ? "" : ` by ${signal}`;
    const stderrDetail = stderr.trim() === "" ? "" : `: ${stderr.trim()}`;
    super(`${reason}${signalDetail}${stderrDetail}`, "process_exit", options);
  }
}

export function exitReason(exitCode: number): string {
  return EXIT_REASONS[exitCode] ?? `Fovea exited with code ${exitCode}`;
}
