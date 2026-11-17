import {
  HAND_FRAME_MAX_AGE_MS,
  HAND_TRACKING_LOSS_MS,
  midpoint,
  pinchMetric,
  type HandFrame,
  type HandObservation,
  type HandPoint
} from './gesture.js'
import { fingerMetrics, metricDistance, type FingerMetrics } from './fingerPose.js'
import { SmoothPoint } from './smoothing.js'

export interface HandPanIntent {
  type: 'pan'
  phase: 'start' | 'move' | 'end' | 'cancel'
  /** Time-integrated camera-normalized displacement: x is mirrored, y is downward.
   * Start/end/cancel have zero displacement. Multiply by viewport size and
   * sensitivity in the host. This remains relative outside the active region. */
  delta: HandPoint
}

const ARM_MS = 100
const START_DISTANCE = 0.012
const UNCERTAIN_POSE_MS = 80
const MAX_FRAME_TRAVEL = 0.12
const FILTER_SCALE = 1000
const FULL_SPEED_DISTANCE = 0.08
const MAX_SPEED = 0.8

const validHand = (hand: HandObservation): boolean =>
  hand.landmarks.length === 21 && hand.landmarks.every((point) => Number.isFinite(point.x) && Number.isFinite(point.y))

/** A relaxed open palm explicitly resumes pointing after a pan stroke.
 * All four fingers must extend beyond their middle joints; no thumb/index
 * pinch is allowed. The test is orientation independent in the image plane. */
export function isOpenHandPose(hand: HandObservation, aspect: number): boolean {
  const fingers = fingerMetrics(hand, aspect)
  return (
    !!fingers &&
    pinchMetric(hand.landmarks, aspect) > 0.55 &&
    fingers.every((finger) => finger.straightness > 0.55 && finger.reach > 0.05)
  )
}

/** Deliberate index+middle extension with ring+pinky folded, with no thumb/index
 * pinch. Distances and bend angles are orientation independent in the image
 * plane. The palm should face the camera; severely edge-on hands are ambiguous. */
export function isTwoFingerPanPose(hand: HandObservation, aspect: number): boolean {
  const fingers = fingerMetrics(hand, aspect)
  if (!fingers || pinchMetric(hand.landmarks, aspect) <= 0.55) return false
  const [index, middle, ring, pinky] = fingers
  return (
    index.straightness > 0.75 &&
    index.reach > 0.15 &&
    middle.straightness > 0.75 &&
    middle.reach > 0.15 &&
    ring.straightness < 0.25 &&
    ring.reach < 0.1 &&
    pinky.straightness < 0.25 &&
    pinky.reach < 0.1
  )
}

/** DOM-free, single-hand velocity panning from a comfortable neutral position.
 *
 * The first recognized pose enters `arming` immediately: the host must suppress
 * pointer/dwell/pinch processing in both arming and panning, including the frame
 * that ends panning. Holding the pose for 100ms AND moving 0.012 camera widths
 * starts navigation, with no initial jump. A small offset controls speed; holding
 * that offset keeps moving without further reach. Returning to neutral stops.
 * Folding either extended finger stops
 * immediately, so a user can reposition and repeat without moving the canvas.
 * Once panning, relaxed finger angles remain valid; an ambiguous sample pauses
 * movement for up to 80ms and rebases on recovery, with no catchup displacement.
 *
 * Only one visible hand is allowed. A new second hand, tracking loss, a large
 * discontinuity or a gap of 400ms cancels the stroke. Fresh frames after a short
 * stall over 150ms but under 400ms rebase the palm while retaining the stroke. Rejected stale or
 * reordered frames do not mutate state or refresh a watchdog; the host
 * must call abort() if fresh frames stop arriving, on pause, or on mode changes.
 */
export class HandPanGesture {
  phase: 'idle' | 'arming' | 'panning' = 'idle'
  private lastTs = -Infinity
  private armedAt = 0
  private origin: HandPoint = { x: 0, y: 0 }
  private wrist: HandPoint | null = null
  private uncertainSince: number | null = null
  private filter = new SmoothPoint()

  constructor(readonly gen: number) {}

  abort(): HandPanIntent[] {
    const intents = this.phase === 'panning' ? [this.intent('cancel')] : []
    this.reset()
    return intents
  }

  step(frame: HandFrame, now: number): { accepted: boolean; intents: HandPanIntent[] } {
    if (
      frame.gen !== this.gen ||
      !Number.isFinite(now) ||
      !Number.isFinite(frame.wallTs) ||
      !Number.isFinite(frame.ts) ||
      now - frame.wallTs > HAND_FRAME_MAX_AGE_MS ||
      frame.wallTs > now ||
      frame.wallTs <= this.lastTs ||
      !Number.isFinite(frame.aspect) ||
      frame.aspect <= 0
    )
      return { accepted: false, intents: [] }

    const gap = frame.wallTs - this.lastTs
    const resume = gap > HAND_FRAME_MAX_AGE_MS && gap < HAND_TRACKING_LOSS_MS
    this.lastTs = frame.wallTs
    const intents: HandPanIntent[] = []
    const result = (): { accepted: boolean; intents: HandPanIntent[] } => ({ accepted: true, intents })
    if (Number.isFinite(gap) && gap >= HAND_TRACKING_LOSS_MS) {
      intents.push(...this.abort())
      return result()
    }
    if (frame.hands.length !== 1 || !validHand(frame.hands[0])) {
      intents.push(...this.abort())
      return result()
    }

    const hand = frame.hands[0]
    const wrist = hand.landmarks[0]
    if (!resume && this.wrist && metricDistance(wrist, this.wrist, frame.aspect) > MAX_FRAME_TRAVEL) {
      intents.push(...this.abort())
      // A tracking jump cannot begin another stroke in the same frame.
      return result()
    }
    const palm = midpoint(wrist, hand.landmarks[9])
    if (this.uncertainSince !== null && frame.wallTs - this.uncertainSince > UNCERTAIN_POSE_MS) {
      intents.push(this.intent('end'))
      this.reset()
      return result()
    }
    const strictPose = isTwoFingerPanPose(hand, frame.aspect)
    if (!strictPose) {
      const fingers = fingerMetrics(hand, frame.aspect)
      const metric = pinchMetric(hand.landmarks, frame.aspect)
      const extended = (finger: FingerMetrics): boolean => finger.straightness > 0.5 && finger.reach > 0.05
      const folded = (finger: FingerMetrics): boolean => finger.straightness < 0.25 && finger.reach < 0.1
      const explicitLift =
        !!fingers && (metric < 0.35 || folded(fingers[0]) || folded(fingers[1]) || isOpenHandPose(hand, frame.aspect))
      const relaxedPose =
        !!fingers &&
        metric > 0.4 &&
        extended(fingers[0]) &&
        extended(fingers[1]) &&
        fingers[2].straightness < 0.55 &&
        fingers[2].reach < 0.2 &&
        fingers[3].straightness < 0.55 &&
        fingers[3].reach < 0.2
      if (this.phase !== 'panning' || explicitLift) {
        if (this.phase === 'panning') intents.push(this.intent('end'))
        this.reset()
        return result()
      }
      if (!relaxedPose) {
        this.uncertainSince ??= frame.wallTs
        this.rebase(palm, wrist, frame.wallTs)
        if (frame.wallTs - this.uncertainSince >= UNCERTAIN_POSE_MS) {
          intents.push(this.intent('end'))
          this.reset()
        }
        return result()
      }
    }
    if ((resume && this.phase !== 'idle') || this.uncertainSince !== null) {
      this.rebase(palm, wrist, frame.wallTs)
      this.uncertainSince = null
      if (this.phase === 'arming') {
        this.origin = palm
        this.armedAt = frame.wallTs
      }
      return result()
    }
    const filtered = this.filter.filter({ x: palm.x * FILTER_SCALE, y: palm.y * FILTER_SCALE }, frame.wallTs)
    const point = { x: filtered.x / FILTER_SCALE, y: filtered.y / FILTER_SCALE }
    this.wrist = wrist
    if (this.phase === 'idle') {
      this.phase = 'arming'
      this.armedAt = frame.wallTs
      this.origin = point
      return result()
    }
    if (this.phase === 'arming') {
      if (frame.wallTs - this.armedAt >= ARM_MS && metricDistance(point, this.origin, frame.aspect) >= START_DISTANCE) {
        this.phase = 'panning'
        intents.push(this.intent('start'))
      }
      return result()
    }
    if (metricDistance(palm, this.origin, frame.aspect) <= START_DISTANCE) {
      this.filter.reset()
      this.filter.filter({ x: palm.x * FILTER_SCALE, y: palm.y * FILTER_SCALE }, frame.wallTs)
      intents.push(this.intent('move'))
      return result()
    }
    const offset = { x: this.origin.x - point.x, y: (point.y - this.origin.y) / frame.aspect }
    const travel = Math.hypot(offset.x, offset.y)
    const speed = Math.min(1, Math.max(0, (travel - START_DISTANCE) / (FULL_SPEED_DISTANCE - START_DISTANCE))) ** 2
    // Fresh samples alone advance the canvas. Never integrate a stalled interval.
    const step = (MAX_SPEED * speed * Math.min(gap, 80)) / 1000 / Math.max(travel, 0.0001)
    const delta = { x: offset.x * step, y: offset.y * step * frame.aspect }
    intents.push({ type: 'pan', phase: 'move', delta })
    return result()
  }

  private intent(phase: HandPanIntent['phase']): HandPanIntent {
    return { type: 'pan', phase, delta: { x: 0, y: 0 } }
  }

  private rebase(palm: HandPoint, wrist: HandPoint, now: number): void {
    this.filter.reset()
    this.filter.filter({ x: palm.x * FILTER_SCALE, y: palm.y * FILTER_SCALE }, now)
    this.origin = palm
    this.wrist = wrist
  }

  private reset(): void {
    this.phase = 'idle'
    this.wrist = null
    this.uncertainSince = null
    this.filter.reset()
  }
}
