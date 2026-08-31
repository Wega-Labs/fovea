import type {
  CalibrationCue,
  CalibrationDone,
  CalibrationWarning,
  FoveaEvent,
} from "./generated/protocol.js";

export * from "./generated/protocol.js";

export interface FoveaHello {
  readonly type: "hello";
  readonly protocol: string;
  readonly fovea: string;
  readonly backend: string;
  readonly coordinate_space: "display_normalized";
  readonly indicator_required: true;
  readonly capabilities: ReadonlyArray<string>;
}

export interface CalibrationTarget {
  readonly label: string;
  readonly x: number;
  readonly y: number;
}

export interface TargetRect {
  readonly id: string;
  readonly x: number;
  readonly y: number;
  readonly w: number;
  readonly h: number;
}

export interface Bounds {
  readonly x: number;
  readonly y: number;
  readonly width: number;
  readonly height: number;
}

export interface ClientPoint {
  readonly x: number;
  readonly y: number;
}

export interface ReconnectOptions {
  readonly initialDelayMs?: number;
  readonly maxDelayMs?: number;
  readonly factor?: number;
  readonly maxAttempts?: number;
}

export interface SpawnFoveaOptions {
  /** Path to the `fovea` executable. Defaults to `fovea`. */
  readonly binary?: string;
  /** Python executable used as `<python> -m fovea`; mutually exclusive with binary. */
  readonly python?: string;
  /** Arguments inserted before `run --ndjson`, useful for launchers such as `uv run fovea`. */
  readonly prefixArgs?: ReadonlyArray<string>;
  /** Additional arguments appended to `fovea run --ndjson`. */
  readonly args?: ReadonlyArray<string>;
  readonly camera?: number;
  readonly backend?: string;
  readonly cwd?: string;
  readonly env?: Readonly<NodeJS.ProcessEnv>;
  readonly startTimeoutMs?: number;
  readonly maxBufferBytes?: number;
  readonly maxGazeHz?: number;
  readonly reconnect?: boolean | ReconnectOptions;
}

export interface RunCalibrationOptions {
  readonly targets?: ReadonlyArray<CalibrationTarget>;
  readonly onCue: (cue: CalibrationCue) => void | Promise<void>;
  readonly onWarning?: (warning: CalibrationWarning) => void;
  readonly timeoutMs?: number;
  readonly signal?: AbortSignal;
}

export type FoveaClientStatus =
  | "starting"
  | "running"
  | "reconnecting"
  | "closing"
  | "closed";

export interface MalformedLineNotice {
  readonly line: string;
  readonly error: Error;
}

export interface ReconnectNotice {
  readonly attempt: number;
  readonly delayMs: number;
  readonly error: Error;
}

export interface UnknownEventNotice {
  readonly type: string;
  readonly value: Readonly<Record<string, unknown>>;
}

type ProtocolEventMap = {
  [Kind in FoveaEvent["type"]]: [event: Extract<FoveaEvent, { type: Kind }>];
};

export type FoveaClientEventMap = ProtocolEventMap & {
  readonly hello: [hello: FoveaHello];
  readonly event: [event: FoveaEvent];
  readonly malformedLine: [notice: MalformedLineNotice];
  readonly processError: [error: Error];
  readonly reconnecting: [notice: ReconnectNotice];
  readonly status: [status: FoveaClientStatus];
  readonly unknownEvent: [notice: UnknownEventNotice];
};

export type CalibrationResult = CalibrationDone;
