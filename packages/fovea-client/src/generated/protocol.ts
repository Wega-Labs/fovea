// Generated from schema/fovea-protocol-v1.json. Do not edit by hand.

export interface GazePoint {
  readonly type: "gaze_point";
  readonly x: number;
  readonly y: number;
  readonly confidence: number;
  readonly timestamp_ns: number;
  readonly target_id?: string | null;
  readonly snapped_x?: number | null;
  readonly snapped_y?: number | null;
}

export interface TargetEnter {
  readonly type: "target_enter";
  readonly id: string;
  readonly timestamp_ns: number;
}

export interface TargetLeave {
  readonly type: "target_leave";
  readonly id: string;
  readonly timestamp_ns: number;
}

export interface DwellProgress {
  readonly type: "dwell_progress";
  readonly id: string;
  readonly progress: number;
  readonly timestamp_ns: number;
}

export interface Dwell {
  readonly type: "dwell";
  readonly id: string;
  readonly timestamp_ns: number;
}

export interface Fixation {
  readonly type: "fixation";
  readonly x: number;
  readonly y: number;
  readonly duration_ms: number;
  readonly confidence: number;
  readonly timestamp_ns: number;
}

export interface Blink {
  readonly type: "blink";
  readonly eye: "left" | "right" | "both";
  readonly duration_ms: number;
  readonly confidence: number;
  readonly timestamp_ns: number;
}

export interface Gesture {
  readonly type: "gesture";
  readonly kind: string;
  readonly phase: "started" | "updated" | "ended" | "cancelled";
  readonly confidence: number;
  readonly timestamp_ns: number;
}

export interface Manipulation {
  readonly type: "manipulation";
  readonly target_id: string;
  readonly phase: "started" | "updated" | "ended" | "cancelled";
  readonly delta_x: number;
  readonly delta_y: number;
  readonly scale: number;
  readonly rotation_degrees: number;
  readonly confidence: number;
  readonly timestamp_ns: number;
}

export interface TrackingState {
  readonly type: "tracking_state";
  readonly status: "active" | "uncertain" | "lost";
  readonly confidence: number;
  readonly timestamp_ns: number;
  readonly detail?: string;
}

export interface CalibrationCue {
  readonly type: "calibration_cue";
  readonly label: string;
  readonly x: number;
  readonly y: number;
  readonly index: number;
  readonly total: number;
  readonly samples: number;
  readonly needed: number;
  readonly instruction: string;
  readonly timestamp_ns: number;
}

export interface CalibrationWarning {
  readonly type: "calibration_warning";
  readonly message: string;
  readonly coverage: number;
  readonly timestamp_ns: number;
}

export interface CalibrationDone {
  readonly type: "calibration_done";
  readonly n_points: number;
  readonly coverage: number;
  readonly loo_error: number;
  readonly timestamp_ns: number;
}

export interface CalibrationUpdated {
  readonly type: "calibration_updated";
  readonly n: number;
  readonly loo_error: number;
  readonly timestamp_ns: number;
}

export interface GazeTestDone {
  readonly type: "gaze_test_done";
  readonly n_points: number;
  readonly mean_error: number;
  readonly median_error: number;
  readonly max_error: number;
  readonly points: ReadonlyArray<{
    readonly expected_x: number;
    readonly expected_y: number;
    readonly predicted_x: number;
    readonly predicted_y: number;
    readonly error: number;
  }>;
  readonly timestamp_ns: number;
}

export interface Diagnostics {
  readonly type: "diagnostics";
  readonly fps: number;
  readonly latency_ms: number;
  readonly face_width: number;
  readonly yaw_deg: number;
  readonly pitch_deg: number;
  readonly timestamp_ns: number;
}

export interface CalibrateCommand {
  readonly cmd: "calibrate";
  readonly targets?: ReadonlyArray<{
    readonly label: string;
    readonly x: number;
    readonly y: number;
  }> | null;
}

export interface TestCommand {
  readonly cmd: "test";
  readonly targets?: ReadonlyArray<{
    readonly label: string;
    readonly x: number;
    readonly y: number;
  }> | null;
}

export interface TargetsCommand {
  readonly cmd: "targets";
  readonly items: ReadonlyArray<{
    readonly id: string;
    readonly x: number;
    readonly y: number;
    readonly w: number;
    readonly h: number;
  }>;
  readonly space: "display_normalized";
}

export interface ObserveCommand {
  readonly cmd: "observe";
  readonly x: number;
  readonly y: number;
  readonly weight?: number;
  readonly timestamp_ns?: number | null;
}

export interface PauseCommand {
  readonly cmd: "pause";
}

export interface ResumeCommand {
  readonly cmd: "resume";
}

export interface QuitCommand {
  readonly cmd: "quit";
}

export type FoveaEvent =
  | GazePoint
  | TargetEnter
  | TargetLeave
  | DwellProgress
  | Dwell
  | Fixation
  | Blink
  | Gesture
  | Manipulation
  | TrackingState
  | CalibrationCue
  | CalibrationWarning
  | CalibrationDone
  | CalibrationUpdated
  | GazeTestDone
  | Diagnostics;

export type FoveaCommand =
  | CalibrateCommand
  | TestCommand
  | TargetsCommand
  | ObserveCommand
  | PauseCommand
  | ResumeCommand
  | QuitCommand;

export const FOVEA_EVENT_TYPES = [
  "gaze_point",
  "target_enter",
  "target_leave",
  "dwell_progress",
  "dwell",
  "fixation",
  "blink",
  "gesture",
  "manipulation",
  "tracking_state",
  "calibration_cue",
  "calibration_warning",
  "calibration_done",
  "calibration_updated",
  "gaze_test_done",
  "diagnostics",
] as const;

export const FOVEA_COMMAND_TYPES = [
  "calibrate",
  "test",
  "targets",
  "observe",
  "pause",
  "resume",
  "quit",
] as const;
