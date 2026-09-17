import { fingerMetrics } from './fingerPose.js'
import { SmoothPoint } from './smoothing.js'

/** Freshness protects against replay; loss allows a brief inference stall. */
export const HAND_FRAME_MAX_AGE_MS = 150
export const HAND_TRACKING_LOSS_MS = 400

const PINCH_CLOSE = 0.35
const PINCH_RELEASE = 0.55
const PINCH_CONFIRM_MS = 40
const PINCH_RELEASE_MS = 60
const DRAG_SLOP_PX = 4
const CURL_CONFIRM_MS = 60
const CURL_TIMEOUT_MS = 1500
const CURL_MAX_TRAVEL = 0.12
export interface HandPoint {
  x: number
  y: number
}

export interface HandObservation {
  landmarks: HandPoint[]
  handedness: string
}

export interface HandFrame {
  gen: number
  wallTs: number
  ts: number
  aspect: number
  hands: HandObservation[]
}

export type HandIntent =
  | ({ type: 'move' | 'down' | 'up' | 'click' } & HandPoint)
  | { type: 'cancel' }
  | { type: 'scale'; phase: 'start' | 'move' | 'end' | 'cancel'; points: [HandPoint, HandPoint] }

export type HandInteractionMode = 'pinch' | 'curl'

type Phase = 'idle' | 'armed' | 'curling' | 'pending' | 'dragging' | 'scaling' | 'cooldown'
interface Track {
  id: number
  wrist: HandPoint
  handedness: string
  tip: HandPoint
  raw: HandPoint
  rawPalm: HandPoint
  palm: HandPoint
  filter: SmoothPoint
  palmFilter: SmoothPoint
  point: HandPoint
  metric: number
  closedSince: number | null
  openSince: number | null
  fingers: ReturnType<typeof fingerMetrics>
  closed: boolean
  open: boolean
}

const clamp = (value: number, min: number, max: number): number => Math.min(max, Math.max(min, value))
export const distance = (a: HandPoint, b: HandPoint): number => Math.hypot(a.x - b.x, a.y - b.y)
export const midpoint = (a: HandPoint, b: HandPoint): HandPoint => ({ x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 })

export interface HandRegion {
  left: number
  top: number
  right: number
  bottom: number
}
export const DEFAULT_HAND_REGION: Readonly<HandRegion> = Object.freeze({ left: 0.1, top: 0.1, right: 0.9, bottom: 0.9 })
export function mapHandPoint(
  tip: HandPoint,
  width: number,
  height: number,
  region: HandRegion = DEFAULT_HAND_REGION
): HandPoint {
  return {
    x: clamp((region.right - tip.x) / (region.right - region.left), 0, 1) * (width - 1),
    y: clamp((tip.y - region.top) / (region.bottom - region.top), 0, 1) * (height - 1)
  }
}

export function pinchMetric(landmarks: HandPoint[], aspect: number): number {
  const length = (a: HandPoint, b: HandPoint): number => Math.hypot(a.x - b.x, (a.y - b.y) / aspect)
  return length(landmarks[4], landmarks[8]) / Math.max(length(landmarks[0], landmarks[9]), 0.0001)
}

export function scaleViewport(
  start: { x: number; y: number; zoom: number },
  anchor: HandPoint,
  startDistance: number,
  currentDistance: number
): { x: number; y: number; zoom: number } {
  const zoom = clamp((start.zoom * currentDistance) / Math.max(1, startDistance), 0.05, 2)
  return {
    x: anchor.x - ((anchor.x - start.x) / start.zoom) * zoom,
    y: anchor.y - ((anchor.y - start.y) / start.zoom) * zoom,
    zoom
  }
}

/** Deterministic, DOM-free gesture reducer. Only the controller turns scale intents into presses. */
export class HandGesture {
  phase: Phase = 'idle'
  private tracks: Track[] = []
  private nextId = 0
  private primary: number | null = null
  private secondary: number | null = null
  private lastTs = -Infinity
  private origin: HandPoint = { x: 0, y: 0 }
  private pinchOrigin: HandPoint = { x: 0, y: 0 }
  private current: HandPoint = { x: 0, y: 0 }
  private scalePoints: [HandPoint, HandPoint] = [this.origin, this.origin]
  private scaleOrigins: [HandPoint, HandPoint] = [this.origin, this.origin]
  private scalePalms: [HandPoint, HandPoint] = [this.origin, this.origin]
  private closing = false
  private curlReady = false
  private curlOthersOpen = false
  private curlStartedAt = 0
  private curlClosedAt: number | null = null
  private curlOpenedAt: number | null = null
  private curlConfirmed = false
  private curlWrist: HandPoint = this.origin

  constructor(
    readonly gen: number,
    private readonly region: HandRegion = DEFAULT_HAND_REGION,
    readonly interaction: HandInteractionMode = 'pinch'
  ) {
    if (
      !Object.values(region).every(Number.isFinite) ||
      region.left < 0 ||
      region.top < 0 ||
      region.right > 1 ||
      region.bottom > 1 ||
      region.right - region.left < 0.15 ||
      region.bottom - region.top < 0.15
    )
      throw new Error('Invalid hand active region')
  }

  get sample(): { point: HandPoint; palm: HandPoint; tip: HandPoint; pinching: boolean } | null {
    const primary = this.tracks.find((track) => track.id === this.primary)
    return primary
      ? { point: primary.point, palm: primary.palm, tip: primary.tip, pinching: primary.metric < 0.5 }
      : null
  }

  abort(): HandIntent[] {
    const intents: HandIntent[] =
      this.phase === 'dragging'
        ? [{ type: 'cancel' }]
        : this.phase === 'scaling'
          ? [{ type: 'scale', phase: 'cancel', points: this.scalePoints }]
          : []
    this.phase = 'idle'
    this.tracks = []
    this.primary = null
    this.secondary = null
    this.closing = false
    this.curlReady = false
    return intents
  }

  /** Rejected frames must not reset the controller's watchdog. */
  step(
    frame: HandFrame,
    now: number,
    width: number,
    height: number,
    target?: HandPoint
  ): { accepted: boolean; intents: HandIntent[] } {
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
    const interrupted = gap >= HAND_TRACKING_LOSS_MS ? this.abort() : []
    const resume = gap > HAND_FRAME_MAX_AGE_MS && gap < HAND_TRACKING_LOSS_MS
    this.lastTs = frame.wallTs
    const hands = frame.hands
      .slice(0, 2)
      .filter(
        (hand) =>
          hand.landmarks.length === 21 &&
          hand.landmarks.every((point) => Number.isFinite(point.x) && Number.isFinite(point.y))
      )
    const pairs = this.tracks
      .flatMap((track) =>
        hands.map((hand, index) => ({
          track,
          index,
          distance: Math.hypot(
            track.wrist.x - hand.landmarks[0].x,
            (track.wrist.y - hand.landmarks[0].y) / frame.aspect
          ),
          tie: track.handedness === hand.handedness ? 0 : 1
        }))
      )
      .filter((pair) => pair.distance <= 0.25)
      .sort((a, b) => a.distance - b.distance || a.tie - b.tie)
    const matched = new Map<number, Track>()
    const used = new Set<number>()
    for (const pair of pairs) {
      if (!matched.has(pair.index) && !used.has(pair.track.id)) {
        matched.set(pair.index, pair.track)
        used.add(pair.track.id)
      }
    }
    this.tracks = hands.map((hand, index) => {
      const raw = mapHandPoint(hand.landmarks[8], width, height, this.region)
      const old = matched.get(index)
      const filter = old?.filter ?? new SmoothPoint()
      const palmFilter = old?.palmFilter ?? new SmoothPoint()
      if (resume) {
        filter.reset()
        palmFilter.reset()
      }
      const palm = midpoint(hand.landmarks[0], hand.landmarks[9])
      // Reach is calibrated from the fingertip. The palm often lies outside
      // that region, especially with compact reach; clamping it would create
      // a dead zone in relative dragging and relaxed-hand placement.
      const rawPalm = {
        x: ((this.region.right - palm.x) / (this.region.right - this.region.left)) * (width - 1),
        y: ((palm.y - this.region.top) / (this.region.bottom - this.region.top)) * (height - 1)
      }
      const metric = pinchMetric(hand.landmarks, frame.aspect)
      const closedSince = metric < PINCH_CLOSE ? (resume ? frame.wallTs : (old?.closedSince ?? frame.wallTs)) : null
      const openSince = metric > PINCH_RELEASE ? (resume ? frame.wallTs : (old?.openSince ?? frame.wallTs)) : null
      return {
        id: old?.id ?? ++this.nextId,
        wrist: hand.landmarks[0],
        handedness: hand.handedness,
        raw,
        rawPalm,
        tip: hand.landmarks[8],
        metric,
        fingers: this.interaction === 'curl' ? fingerMetrics(hand, frame.aspect) : null,
        point: filter.filter(raw, frame.wallTs),
        filter,
        palmFilter,
        palm: palmFilter.filter(rawPalm, frame.wallTs),
        closedSince,
        openSince,
        closed: closedSince !== null && frame.wallTs - closedSince >= PINCH_CONFIRM_MS,
        open: openSince !== null && frame.wallTs - openSince >= PINCH_RELEASE_MS
      }
    })
    const result = (intents: HandIntent[] = []): { accepted: boolean; intents: HandIntent[] } => ({
      accepted: true,
      intents: [...interrupted, ...intents]
    })
    if (!this.tracks.length) return result(this.abort())
    if (this.phase === 'cooldown') {
      if (!this.tracks.every((track) => track.open)) return result()
      this.phase = 'idle'
    }
    if (this.phase === 'idle') {
      const armed = this.tracks.find((track) => track.metric > 0.5)
      if (!armed) return result()
      this.primary = armed.id
      this.phase = 'armed'
    }
    const primary = this.tracks.find((track) => track.id === this.primary)
    if (!primary) return result(this.abort())
    if (resume && this.closing) {
      // Resume from the latest palm sample without replaying travel accumulated
      // while inference was stalled. Keep the current press/placement alive.
      if (this.phase === 'dragging') this.origin = this.current
      this.pinchOrigin = primary.palm
      if (this.phase === 'scaling') {
        const secondary = this.tracks.find((track) => track.id === this.secondary)
        if (!secondary) return result(this.abort())
        this.scaleOrigins = this.scalePoints
        this.scalePalms = [primary.palm, secondary.palm]
      }
    }
    const index = primary.fingers?.[0]
    const extended = !!index && index.straightness > 0.75 && index.reach > 0.15
    const curled = !!index && index.straightness < 0.25 && index.reach < 0.1
    const otherFingerOpen = primary.fingers
      ?.slice(1)
      .some((finger) => finger.straightness > 0.55 && finger.reach > 0.05)
    if (this.phase === 'curling') {
      // A click belongs to one continuous pointing hand. Never click after a
      // tracking gap, whole-hand closing, long hold, or deliberate departure.
      // Fingers already folded during pointing can stay relaxed throughout the click.
      if (
        resume ||
        (this.curlOthersOpen && !otherFingerOpen) ||
        frame.wallTs - this.curlStartedAt > CURL_TIMEOUT_MS ||
        Math.hypot(primary.wrist.x - this.curlWrist.x, (primary.wrist.y - this.curlWrist.y) / frame.aspect) >
          CURL_MAX_TRAVEL
      ) {
        this.phase = 'cooldown'
        return result()
      }
      if (primary.metric < PINCH_CLOSE || this.tracks.length > 1) {
        // A second hand cancels the click but can still join a two-hand grip.
        // Closing toward the thumb is a grip, even if the index bent first.
        this.phase = 'armed'
        this.closing = false
      } else {
        this.curlClosedAt = curled ? (this.curlClosedAt ?? frame.wallTs) : null
        if (this.curlClosedAt !== null && frame.wallTs - this.curlClosedAt >= CURL_CONFIRM_MS) this.curlConfirmed = true
        this.curlOpenedAt = extended ? (this.curlOpenedAt ?? frame.wallTs) : null
        if (this.curlOpenedAt !== null && frame.wallTs - this.curlOpenedAt >= CURL_CONFIRM_MS) {
          this.phase = 'cooldown'
          primary.filter.reset()
          return result(this.curlConfirmed ? [{ type: 'click', ...this.origin }] : [])
        }
        return result()
      }
    }
    if (this.phase === 'armed') {
      if (this.tracks.length === 1 && extended && primary.metric > PINCH_RELEASE) {
        this.curlReady = true
        this.curlOthersOpen = !!otherFingerOpen
      }
      if (this.curlReady && index && index.straightness < 0.65 && this.tracks.length === 1 && primary.metric > 0.55) {
        this.origin = target ?? this.current
        this.current = this.origin
        this.curlWrist = primary.wrist
        this.curlStartedAt = frame.wallTs
        this.curlClosedAt = curled ? frame.wallTs : null
        this.curlOpenedAt = null
        this.curlConfirmed = false
        this.curlReady = false
        this.phase = 'curling'
        return result()
      }
      if (primary.metric >= PINCH_CLOSE) {
        this.closing = false
        this.current = target ?? primary.point
        return result([{ type: 'move', ...this.current }])
      }
      // Capture the previously displayed aim before curling the finger moves its tip.
      if (!this.closing) {
        this.origin = target ?? this.current
        // Begin tracking palm travel with the first close candidate, so pinch
        // confirmation and movement slop run together instead of stacking.
        primary.palmFilter.reset()
        primary.palm = primary.palmFilter.filter(primary.rawPalm, frame.wallTs)
        this.pinchOrigin = primary.palm
        this.closing = true
        this.curlReady = false
      }
      if (!primary.closed) return result()
      primary.filter.reset()
      primary.point = primary.filter.filter(primary.raw, frame.wallTs)
      this.current = this.origin
      this.phase = 'pending'
    }
    if ((this.phase === 'pending' || this.phase === 'dragging') && primary.metric < PINCH_RELEASE) {
      const secondary = this.tracks.find((track) => track.id !== primary.id && track.closed)
      if (secondary) {
        const dragging = this.phase === 'dragging'
        this.secondary = secondary.id
        // Pointing reach can clamp both fingertips to the same screen edge.
        // Scale geometry uses the full, unbounded camera span instead.
        const region = DEFAULT_HAND_REGION
        this.scalePoints = [primary, secondary].map((track) => ({
          x: ((region.right - track.tip.x) / (region.right - region.left)) * (width - 1),
          y: ((track.tip.y - region.top) / (region.bottom - region.top)) * (height - 1)
        })) as [HandPoint, HandPoint]
        this.scaleOrigins = this.scalePoints
        this.scalePalms = [primary.palm, secondary.palm]
        this.phase = 'scaling'
        return result([
          // Cancel a native drag before the host takes its resize handle. This
          // also prevents a dragged connection from being dropped accidentally.
          ...(dragging ? [{ type: 'cancel' } as const] : []),
          { type: 'scale', phase: 'start', points: this.scalePoints }
        ])
      }
    }
    if (this.phase === 'pending') {
      if (primary.open) {
        this.phase = 'cooldown'
        return result(
          this.interaction === 'pinch'
            ? [
                { type: 'down', ...this.origin },
                { type: 'up', ...this.origin }
              ]
            : []
        )
      }
      // Give the other pinch its normal confirmation time instead of letting
      // the first hand's movement turn a two-hand gesture into a drag.
      if (this.tracks.some((track) => track.id !== primary.id && track.metric < PINCH_CLOSE)) return result()
      if (
        primary.metric >= PINCH_CLOSE ||
        distance(primary.rawPalm, this.pinchOrigin) < DRAG_SLOP_PX ||
        distance(primary.palm, this.pinchOrigin) < DRAG_SLOP_PX / 2
      )
        return result()
      this.phase = 'dragging'
      this.current = this.dragPoint(primary.palm)
      return result([
        { type: 'down', ...this.origin },
        { type: 'move', ...this.current }
      ])
    }
    if (this.phase === 'dragging') {
      // The palm remains stable while fingers open. Keep moving through the
      // pinch hysteresis band instead of freezing until the grip tightens.
      if (primary.open) {
        this.phase = 'cooldown'
        return result([{ type: 'up', ...this.current }])
      }
      this.current = this.dragPoint(primary.palm)
      return result([{ type: 'move', ...this.current }])
    }
    if (this.phase === 'scaling') {
      const secondary = this.tracks.find((track) => track.id === this.secondary)
      if (!secondary) return result(this.abort())
      if (primary.open || secondary.open) {
        this.phase = 'cooldown'
        return result([{ type: 'scale', phase: 'end', points: this.scalePoints }])
      }
      // Preserve the initial fingertip anchor while the palms drive span
      // changes, including uncertain grip readings before confirmed release.
      this.scalePoints = [primary, secondary].map((track, index) => ({
        x:
          this.scaleOrigins[index].x +
          ((track.palm.x - this.scalePalms[index].x) * (this.region.right - this.region.left)) /
            (DEFAULT_HAND_REGION.right - DEFAULT_HAND_REGION.left),
        y:
          this.scaleOrigins[index].y +
          ((track.palm.y - this.scalePalms[index].y) * (this.region.bottom - this.region.top)) /
            (DEFAULT_HAND_REGION.bottom - DEFAULT_HAND_REGION.top)
      })) as [HandPoint, HandPoint]
      return result([{ type: 'scale', phase: 'move', points: this.scalePoints }])
    }
    return result()
  }

  private dragPoint(point: HandPoint): HandPoint {
    return { x: this.origin.x + point.x - this.pinchOrigin.x, y: this.origin.y + point.y - this.pinchOrigin.y }
  }
}
