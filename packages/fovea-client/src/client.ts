import { spawn } from "node:child_process";
import type { ChildProcessWithoutNullStreams } from "node:child_process";
import { EventEmitter } from "node:events";

import { AsyncEventSubscription } from "./async-events.js";
import {
  FoveaClientError,
  FoveaClosedError,
  FoveaHandshakeError,
  FoveaProcessError,
} from "./errors.js";
import { NdjsonDecoder } from "./framing.js";
import { FOVEA_EVENT_TYPES } from "./generated/protocol.js";
import type {
  CalibrateCommand,
  CalibrationCue,
  CalibrationDone,
  CalibrationWarning,
  FoveaCommand,
  FoveaEvent,
  GazePoint,
  TargetsCommand,
} from "./generated/protocol.js";
import type {
  CalibrationTarget,
  FoveaClientEventMap,
  FoveaClientStatus,
  FoveaHello,
  ReconnectNotice,
  ReconnectOptions,
  RunCalibrationOptions,
  SpawnFoveaOptions,
  TargetRect,
} from "./types.js";

const PROTOCOL_MAJOR = "1";
const DEFAULT_START_TIMEOUT_MS = 10_000;
const DEFAULT_MAX_BUFFER_BYTES = 1024 * 1024;
const DEFAULT_MAX_GAZE_HZ = 60;
const STDERR_TAIL_LIMIT = 16 * 1024;
const KNOWN_EVENT_TYPES: ReadonlySet<string> = new Set(FOVEA_EVENT_TYPES);

interface NormalizedReconnectOptions {
  readonly initialDelayMs: number;
  readonly maxDelayMs: number;
  readonly factor: number;
  readonly maxAttempts: number;
}

interface NormalizedOptions {
  readonly executable: string;
  readonly executableArgs: ReadonlyArray<string>;
  readonly cwd: string | undefined;
  readonly env: NodeJS.ProcessEnv;
  readonly startTimeoutMs: number;
  readonly maxBufferBytes: number;
  readonly maxGazeHz: number;
  readonly reconnect: NormalizedReconnectOptions | null;
}

export interface FoveaClient {
  on<Kind extends keyof FoveaClientEventMap>(
    eventName: Kind,
    listener: (...args: FoveaClientEventMap[Kind]) => void,
  ): this;
  once<Kind extends keyof FoveaClientEventMap>(
    eventName: Kind,
    listener: (...args: FoveaClientEventMap[Kind]) => void,
  ): this;
  off<Kind extends keyof FoveaClientEventMap>(
    eventName: Kind,
    listener: (...args: FoveaClientEventMap[Kind]) => void,
  ): this;
}

export class FoveaClient extends EventEmitter implements AsyncIterable<FoveaEvent> {
  readonly ready: Promise<FoveaHello>;

  private readonly options: NormalizedOptions;
  private readonly subscriptions = new Set<AsyncEventSubscription<FoveaEvent>>();
  private resolveReady!: (hello: FoveaHello) => void;
  private rejectReady!: (error: Error) => void;
  private readySettled = false;
  private currentStatus: FoveaClientStatus = "starting";
  private child: ChildProcessWithoutNullStreams | undefined;
  private decoder: NdjsonDecoder;
  private sawHello = false;
  private stderrTail = "";
  private userClosed = false;
  private reconnectAttempts = 0;
  private reconnectTimer: NodeJS.Timeout | undefined;
  private startTimer: NodeJS.Timeout | undefined;
  private pendingGaze: GazePoint | undefined;
  private gazeTimer: NodeJS.Timeout | undefined;
  private lastGazeAt = Number.NEGATIVE_INFINITY;
  private calibrationActive = false;
  private terminalError: Error | undefined;

  constructor(options: SpawnFoveaOptions = {}) {
    super();
    this.options = normalizeOptions(options);
    this.decoder = new NdjsonDecoder(this.options.maxBufferBytes);
    this.ready = new Promise((resolve, reject) => {
      this.resolveReady = resolve;
      this.rejectReady = reject;
    });
    queueMicrotask(() => this.launch());
  }

  get status(): FoveaClientStatus {
    return this.currentStatus;
  }

  send(command: FoveaCommand): void {
    const child = this.child;
    if (this.currentStatus !== "running" || child === undefined || !child.stdin.writable) {
      throw new FoveaClosedError();
    }
    child.stdin.write(`${JSON.stringify(command)}\n`);
  }

  calibrate(targets?: ReadonlyArray<CalibrationTarget>): void {
    validateCalibrationTargets(targets);
    const command: CalibrateCommand =
      targets === undefined ? { cmd: "calibrate" } : { cmd: "calibrate", targets };
    this.send(command);
  }

  runGazeTest(targets?: ReadonlyArray<CalibrationTarget>): void {
    validateCalibrationTargets(targets);
    this.send(targets === undefined ? { cmd: "test" } : { cmd: "test", targets });
  }

  setTargets(targets: ReadonlyArray<TargetRect>): void {
    validateTargetRects(targets);
    const command: TargetsCommand = {
      cmd: "targets",
      items: targets,
      space: "display_normalized",
    };
    this.send(command);
  }

  pause(): void {
    this.send({ cmd: "pause" });
  }

  resume(): void {
    this.send({ cmd: "resume" });
  }

  async runCalibration(options: RunCalibrationOptions): Promise<CalibrationDone> {
    await this.ready;
    if (this.calibrationActive) {
      throw new FoveaClientError("a calibration is already running", "calibration_active");
    }
    validateCalibrationTargets(options.targets);
    if (options.signal?.aborted === true) throw abortError(options.signal);
    const timeoutMs = options.timeoutMs ?? 120_000;
    requirePositiveFinite(timeoutMs, "timeoutMs");
    this.calibrationActive = true;

    return new Promise<CalibrationDone>((resolve, reject) => {
      let timer: NodeJS.Timeout | undefined;
      let settled = false;
      const cleanup = (): void => {
        if (timer !== undefined) clearTimeout(timer);
        this.off("calibration_cue", onCue);
        this.off("calibration_warning", onWarning);
        this.off("calibration_done", onDone);
        this.off("processError", onProcessError);
        this.off("reconnecting", onReconnect);
        options.signal?.removeEventListener("abort", onAbort);
        this.calibrationActive = false;
      };
      const fail = (error: Error): void => {
        if (settled) return;
        settled = true;
        cleanup();
        reject(error);
      };
      const onCue = (cue: CalibrationCue): void => {
        try {
          Promise.resolve(options.onCue(cue)).catch((error: unknown) => fail(asError(error)));
        } catch (error: unknown) {
          fail(asError(error));
        }
      };
      const onWarning = (warning: CalibrationWarning): void => {
        try {
          options.onWarning?.(warning);
        } catch (error: unknown) {
          fail(asError(error));
        }
      };
      const onDone = (result: CalibrationDone): void => {
        if (settled) return;
        settled = true;
        cleanup();
        resolve(result);
      };
      const onProcessError = (error: Error): void => fail(error);
      const onReconnect = (notice: ReconnectNotice): void => fail(notice.error);
      const onAbort = (): void => fail(abortError(options.signal));

      this.on("calibration_cue", onCue);
      this.on("calibration_warning", onWarning);
      this.on("calibration_done", onDone);
      this.on("processError", onProcessError);
      this.on("reconnecting", onReconnect);
      options.signal?.addEventListener("abort", onAbort, { once: true });
      if (options.signal?.aborted === true) {
        onAbort();
        return;
      }
      timer = setTimeout(
        () => fail(new FoveaClientError("calibration timed out", "calibration_timeout")),
        timeoutMs,
      );

      try {
        this.calibrate(options.targets);
      } catch (error: unknown) {
        fail(asError(error));
      }
    });
  }

  close(): void {
    if (this.userClosed) return;
    this.userClosed = true;
    this.setStatus("closing");
    this.clearTimers();
    this.clearPendingGaze();
    const child = this.child;
    this.child = undefined;
    if (child !== undefined) {
      try {
        if (child.stdin.writable) child.stdin.write('{"cmd":"quit"}\n');
      } catch {
        // Termination below is authoritative.
      }
      terminateProcessTree(child);
    }
    if (!this.readySettled) {
      this.readySettled = true;
      this.rejectReady(new FoveaClosedError());
    }
    this.setStatus("closed");
    this.endSubscriptions();
  }

  [Symbol.asyncIterator](): AsyncIterator<FoveaEvent> {
    let subscription: AsyncEventSubscription<FoveaEvent>;
    subscription = new AsyncEventSubscription(() => this.subscriptions.delete(subscription));
    this.subscriptions.add(subscription);
    if (this.currentStatus === "closed") subscription.end(this.terminalError);
    return subscription;
  }

  private launch(): void {
    if (this.userClosed) return;
    this.sawHello = false;
    this.stderrTail = "";
    this.decoder = new NdjsonDecoder(this.options.maxBufferBytes);
    let child: ChildProcessWithoutNullStreams;
    try {
      child = spawn(this.options.executable, this.options.executableArgs, {
        ...(this.options.cwd === undefined ? {} : { cwd: this.options.cwd }),
        env: this.options.env,
        stdio: ["pipe", "pipe", "pipe"],
        windowsHide: true,
        detached: process.platform !== "win32",
      });
    } catch (error: unknown) {
      this.handleFailure(
        undefined,
        new FoveaClientError("unable to spawn Fovea", "spawn", { cause: error }),
        false,
      );
      return;
    }
    this.child = child;
    child.stdout.on("data", (chunk: Buffer) => this.handleData(child, chunk));
    child.stderr.on("data", (chunk: Buffer) => this.captureStderr(child, chunk));
    child.once("error", (error: Error) => {
      this.handleFailure(
        child,
        new FoveaClientError("unable to spawn Fovea", "spawn", { cause: error }),
        true,
      );
    });
    child.once("close", (code: number | null, signal: NodeJS.Signals | null) => {
      if (this.child !== child) return;
      try {
        for (const line of this.decoder.finish()) this.handleLine(child, line);
      } catch (error: unknown) {
        this.handleFailure(child, asError(error), false);
        return;
      }
      if (this.child !== child) return;
      this.handleFailure(
        child,
        new FoveaProcessError(code, signal, this.stderrTail),
        false,
      );
    });
    this.startTimer = setTimeout(() => {
      this.handleFailure(
        child,
        new FoveaHandshakeError(
          `Fovea did not send a hello line within ${this.options.startTimeoutMs} ms`,
        ),
        true,
      );
    }, this.options.startTimeoutMs);
  }

  private handleData(child: ChildProcessWithoutNullStreams, chunk: Buffer): void {
    if (this.child !== child) return;
    try {
      for (const line of this.decoder.push(chunk)) {
        this.handleLine(child, line);
        if (this.child !== child) break;
      }
    } catch (error: unknown) {
      this.handleFailure(child, asError(error), true);
    }
  }

  private handleLine(child: ChildProcessWithoutNullStreams, line: string): void {
    if (line.trim() === "") return;
    let value: unknown;
    try {
      value = JSON.parse(line);
    } catch (error: unknown) {
      const parsedError = asError(error);
      if (!this.sawHello) {
        this.handleFailure(
          child,
          new FoveaHandshakeError("Fovea's first stdout line was not valid JSON", {
            cause: parsedError,
          }),
          true,
        );
      } else {
        this.emit("malformedLine", { line, error: parsedError });
      }
      return;
    }

    if (!this.sawHello) {
      let hello: FoveaHello;
      try {
        hello = parseHello(value);
      } catch (error: unknown) {
        this.handleFailure(child, asError(error), true);
        return;
      }
      this.sawHello = true;
      if (this.startTimer !== undefined) clearTimeout(this.startTimer);
      this.startTimer = undefined;
      this.reconnectAttempts = 0;
      this.setStatus("running");
      if (!this.readySettled) {
        this.readySettled = true;
        this.resolveReady(hello);
      }
      this.emit("hello", hello);
      return;
    }

    if (!isRecord(value) || typeof value.type !== "string") {
      this.emit("malformedLine", {
        line,
        error: new TypeError("protocol event must be an object with a string type"),
      });
      return;
    }
    if (!KNOWN_EVENT_TYPES.has(value.type)) {
      this.emit("unknownEvent", { type: value.type, value });
      return;
    }
    this.acceptEvent(value as unknown as FoveaEvent);
  }

  private acceptEvent(event: FoveaEvent): void {
    if (event.type !== "gaze_point") {
      this.dispatchEvent(event);
      return;
    }
    const intervalMs = 1000 / this.options.maxGazeHz;
    const now = Date.now();
    const elapsed = now - this.lastGazeAt;
    if (elapsed >= intervalMs && this.gazeTimer === undefined) {
      this.lastGazeAt = now;
      this.dispatchEvent(event);
      return;
    }
    this.pendingGaze = event;
    if (this.gazeTimer !== undefined) return;
    const delayMs = Math.max(0, intervalMs - elapsed);
    this.gazeTimer = setTimeout(() => {
      this.gazeTimer = undefined;
      const latest = this.pendingGaze;
      this.pendingGaze = undefined;
      if (latest === undefined || this.currentStatus !== "running") return;
      this.lastGazeAt = Date.now();
      this.dispatchEvent(latest);
    }, delayMs);
  }

  private dispatchEvent(event: FoveaEvent): void {
    this.emit("event", event);
    this.emit(event.type, event);
    for (const subscription of this.subscriptions) subscription.push(event);
  }

  private captureStderr(child: ChildProcessWithoutNullStreams, chunk: Buffer): void {
    if (this.child !== child) return;
    this.stderrTail = `${this.stderrTail}${chunk.toString("utf8")}`.slice(-STDERR_TAIL_LIMIT);
  }

  private handleFailure(
    child: ChildProcessWithoutNullStreams | undefined,
    error: Error,
    terminate: boolean,
  ): void {
    if (child !== undefined && this.child !== child) return;
    if (child === undefined && this.child !== undefined) return;
    const activeChild = child ?? this.child;
    this.child = undefined;
    if (this.startTimer !== undefined) clearTimeout(this.startTimer);
    this.startTimer = undefined;
    this.clearPendingGaze();
    if (terminate && activeChild !== undefined) terminateProcessTree(activeChild);
    if (this.userClosed) return;

    const reconnect = this.options.reconnect;
    if (reconnect !== null && this.reconnectAttempts < reconnect.maxAttempts) {
      this.reconnectAttempts += 1;
      const delayMs = Math.min(
        reconnect.maxDelayMs,
        reconnect.initialDelayMs * reconnect.factor ** (this.reconnectAttempts - 1),
      );
      this.setStatus("reconnecting");
      this.emit("reconnecting", {
        attempt: this.reconnectAttempts,
        delayMs,
        error,
      });
      this.reconnectTimer = setTimeout(() => {
        this.reconnectTimer = undefined;
        this.launch();
      }, delayMs);
      return;
    }

    this.terminalError = error;
    if (!this.readySettled) {
      this.readySettled = true;
      this.rejectReady(error);
    }
    this.setStatus("closed");
    this.emit("processError", error);
    this.endSubscriptions(error);
  }

  private clearPendingGaze(): void {
    if (this.gazeTimer !== undefined) clearTimeout(this.gazeTimer);
    this.gazeTimer = undefined;
    this.pendingGaze = undefined;
    this.lastGazeAt = Number.NEGATIVE_INFINITY;
  }

  private clearTimers(): void {
    if (this.startTimer !== undefined) clearTimeout(this.startTimer);
    if (this.reconnectTimer !== undefined) clearTimeout(this.reconnectTimer);
    this.startTimer = undefined;
    this.reconnectTimer = undefined;
  }

  private setStatus(status: FoveaClientStatus): void {
    if (this.currentStatus === status) return;
    this.currentStatus = status;
    this.emit("status", status);
  }

  private endSubscriptions(error?: Error): void {
    for (const subscription of this.subscriptions) subscription.end(error);
    this.subscriptions.clear();
  }
}

export function spawnFovea(options: SpawnFoveaOptions = {}): FoveaClient {
  return new FoveaClient(options);
}

function normalizeOptions(options: SpawnFoveaOptions): NormalizedOptions {
  if (options.binary !== undefined && options.python !== undefined) {
    throw new TypeError("binary and python are mutually exclusive");
  }
  if (
    options.camera !== undefined &&
    (!Number.isSafeInteger(options.camera) || options.camera < 0)
  ) {
    throw new RangeError("camera must be a non-negative safe integer");
  }
  const startTimeoutMs = options.startTimeoutMs ?? DEFAULT_START_TIMEOUT_MS;
  const maxBufferBytes = options.maxBufferBytes ?? DEFAULT_MAX_BUFFER_BYTES;
  const maxGazeHz = options.maxGazeHz ?? DEFAULT_MAX_GAZE_HZ;
  requirePositiveFinite(startTimeoutMs, "startTimeoutMs");
  if (!Number.isSafeInteger(maxBufferBytes) || maxBufferBytes < 1) {
    throw new RangeError("maxBufferBytes must be a positive safe integer");
  }
  requirePositiveFinite(maxGazeHz, "maxGazeHz");

  const executable = options.python ?? options.binary ?? "fovea";
  if (executable.trim() === "") throw new TypeError("binary or python must be non-empty");
  const executableArgs = [...(options.prefixArgs ?? [])];
  if (options.python !== undefined) executableArgs.push("-m", "fovea");
  executableArgs.push("run", "--ndjson");
  if (options.camera !== undefined) executableArgs.push("--camera", String(options.camera));
  if (options.backend !== undefined) {
    if (options.backend.trim() === "") throw new TypeError("backend must be non-empty");
    executableArgs.push("--backend", options.backend);
  }
  executableArgs.push(...(options.args ?? []));

  return {
    executable,
    executableArgs,
    cwd: options.cwd,
    env: { ...process.env, ...options.env },
    startTimeoutMs,
    maxBufferBytes,
    maxGazeHz,
    reconnect: normalizeReconnect(options.reconnect),
  };
}

function normalizeReconnect(
  reconnect: boolean | ReconnectOptions | undefined,
): NormalizedReconnectOptions | null {
  if (reconnect === undefined || reconnect === false) return null;
  const configured = reconnect === true ? {} : reconnect;
  const initialDelayMs = configured.initialDelayMs ?? 250;
  const maxDelayMs = configured.maxDelayMs ?? 4_000;
  const factor = configured.factor ?? 2;
  const maxAttempts = configured.maxAttempts ?? 5;
  requirePositiveFinite(initialDelayMs, "reconnect.initialDelayMs");
  requirePositiveFinite(maxDelayMs, "reconnect.maxDelayMs");
  requirePositiveFinite(factor, "reconnect.factor");
  if (factor < 1) throw new RangeError("reconnect.factor must be at least one");
  if (!Number.isSafeInteger(maxAttempts) || maxAttempts < 1) {
    throw new RangeError("reconnect.maxAttempts must be a positive safe integer");
  }
  if (maxDelayMs < initialDelayMs) {
    throw new RangeError("reconnect.maxDelayMs must be at least initialDelayMs");
  }
  return { initialDelayMs, maxDelayMs, factor, maxAttempts };
}

function parseHello(value: unknown): FoveaHello {
  if (!isRecord(value) || value.type !== "hello") {
    throw new FoveaHandshakeError("Fovea's first stdout object was not hello");
  }
  if (
    typeof value.protocol !== "string" ||
    typeof value.fovea !== "string" ||
    typeof value.backend !== "string" ||
    value.coordinate_space !== "display_normalized" ||
    value.indicator_required !== true ||
    !Array.isArray(value.capabilities) ||
    !value.capabilities.every((item) => typeof item === "string") ||
    value.protocol.trim() === "" ||
    value.fovea.trim() === "" ||
    value.backend.trim() === ""
  ) {
    throw new FoveaHandshakeError("Fovea sent an incomplete or unsafe hello object");
  }
  if (value.protocol.split(".", 1)[0] !== PROTOCOL_MAJOR) {
    throw new FoveaHandshakeError(
      `unsupported Fovea protocol ${value.protocol}; expected major ${PROTOCOL_MAJOR}`,
    );
  }
  return value as unknown as FoveaHello;
}

function validateCalibrationTargets(
  targets: ReadonlyArray<CalibrationTarget> | undefined,
): void {
  if (targets === undefined) return;
  if (targets.length < 5) throw new RangeError("calibration requires at least five targets");
  for (const target of targets) {
    if (
      target.label.trim() === "" ||
      !Number.isFinite(target.x) ||
      !Number.isFinite(target.y) ||
      target.x < 0 ||
      target.x > 1 ||
      target.y < 0 ||
      target.y > 1
    ) {
      throw new RangeError("calibration targets need labels and coordinates within [0, 1]");
    }
  }
}

function validateTargetRects(targets: ReadonlyArray<TargetRect>): void {
  const ids = new Set<string>();
  for (const target of targets) {
    const coordinates = [target.x, target.y, target.w, target.h];
    if (
      target.id.trim() === "" ||
      ids.has(target.id) ||
      !coordinates.every(Number.isFinite) ||
      target.x < 0 ||
      target.y < 0 ||
      target.w <= 0 ||
      target.h <= 0 ||
      target.x + target.w > 1 ||
      target.y + target.h > 1
    ) {
      throw new RangeError("targets need unique IDs and rectangles within [0, 1]");
    }
    ids.add(target.id);
  }
}

function requirePositiveFinite(value: number, name: string): void {
  if (!Number.isFinite(value) || value <= 0) {
    throw new RangeError(`${name} must be finite and greater than zero`);
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function abortError(signal: AbortSignal | undefined): Error {
  if (signal?.reason instanceof Error) return signal.reason;
  return new DOMException("calibration aborted", "AbortError");
}

function asError(value: unknown): Error {
  return value instanceof Error ? value : new Error(String(value));
}

function terminateProcessTree(child: ChildProcessWithoutNullStreams): void {
  const pid = child.pid;
  if (process.platform === "win32" && pid !== undefined) {
    const killer = spawn("taskkill", ["/pid", String(pid), "/T", "/F"], {
      stdio: "ignore",
      windowsHide: true,
    });
    killer.on("error", () => child.kill());
    killer.unref();
    const fallbackTimer = setTimeout(() => {
      if (child.exitCode === null && child.signalCode === null) child.kill();
    }, 500);
    fallbackTimer.unref();
  } else if (pid !== undefined && pid !== process.pid) {
    try {
      process.kill(-pid, "SIGTERM");
    } catch {
      child.kill("SIGTERM");
    }
  } else {
    child.kill("SIGTERM");
  }
  if (process.platform !== "win32" && pid !== undefined && pid !== process.pid) {
    const forceTimer = setTimeout(() => {
      if (child.exitCode !== null || child.signalCode !== null) return;
      try {
        process.kill(-pid, "SIGKILL");
      } catch {
        child.kill("SIGKILL");
      }
    }, 500);
    forceTimer.unref();
  }
}
