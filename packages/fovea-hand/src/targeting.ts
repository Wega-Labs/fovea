import { distance, type HandPoint } from './gesture.js'

export interface HandRect {
  x: number
  y: number
  width: number
  height: number
}
export interface HandTarget {
  /** Stable for the life of this action; use a new ID when its meaning changes. */
  id: string
  label: string
  rect: HandRect
  point?: HandPoint
  activation: 'dwell' | 'manual'
}
export interface TargetFeedback {
  target: HandTarget | null
  point: HandPoint
  progress: number
  settled: boolean
  activate: HandTarget | null
  ambiguous: HandTarget[]
}
export function rectDistance(point: HandPoint, rect: HandRect): number {
  return Math.hypot(
    Math.max(rect.x - point.x, 0, point.x - rect.x - rect.width),
    Math.max(rect.y - point.y, 0, point.y - rect.y - rect.height)
  )
}
export function targetPoint(target: HandTarget): HandPoint {
  return target.point ?? { x: target.rect.x + target.rect.width / 2, y: target.rect.y + target.rect.height / 2 }
}
const contains = (outer: HandRect, inner: HandRect): boolean =>
  inner.x >= outer.x &&
  inner.y >= outer.y &&
  inner.x + inner.width <= outer.x + outer.width &&
  inner.y + inner.height <= outer.y + outer.height

/** Target assistance and dwell are independent of DOM, React, camera, and native input. */
export class HandTargeting {
  private current: HandTarget | null = null
  private anchor: HandPoint | null = null
  private since = 0
  private ambiguity = ''
  private lastTime = -Infinity
  private blocked: HandRect | null = null
  private cooldownUntil = 0
  readonly dwellMs: number
  readonly radius: number
  readonly strength: number

  constructor({
    dwellMs = 700,
    radius = 28,
    strength = 0.65
  }: { dwellMs?: number; radius?: number; strength?: number } = {}) {
    if (
      !Number.isFinite(dwellMs) ||
      dwellMs < 300 ||
      dwellMs > 2000 ||
      !Number.isFinite(radius) ||
      radius < 0 ||
      radius > 100 ||
      !Number.isFinite(strength) ||
      strength < 0 ||
      strength > 1
    )
      throw new Error('Invalid hand targeting options')
    this.dwellMs = dwellMs
    this.radius = radius
    this.strength = strength
  }
  get target(): HandTarget | null {
    return this.current
  }

  /** Broad-phase discovery must include the current target's retention margin. */
  get searchRadius(): number {
    return this.strength > 0 && this.radius > 0 ? this.radius + 8 : 0
  }

  reset(): void {
    this.current = null
    this.anchor = null
    this.since = 0
    this.ambiguity = ''
  }

  /** Prevent a held aim or newly revealed control from activating again after a click. */
  block(target: HandTarget, now: number): void {
    const point = this.anchor ?? targetPoint(target)
    this.blocked = { x: point.x - 12, y: point.y - 12, width: 24, height: 24 }
    this.cooldownUntil = now + 450
    this.reset()
  }

  update(point: HandPoint, now: number, targets: readonly HandTarget[], dwellEnabled = true): TargetFeedback {
    const result: TargetFeedback = { target: null, point, progress: 0, settled: false, activate: null, ambiguous: [] }
    if (!Number.isFinite(now) || !Number.isFinite(point.x) || !Number.isFinite(point.y) || now <= this.lastTime)
      return result
    if (now - this.lastTime > 150) this.reset()
    this.lastTime = now
    const magnetic = this.strength > 0 && this.radius > 0
    const radius = magnetic ? this.radius : 0
    const retention = this.searchRadius - radius
    if (this.blocked && rectDistance(point, this.blocked) > radius + retention) this.blocked = null
    const ranked = targets
      .filter(
        (target) =>
          target.id &&
          target.label &&
          Object.values(target.rect).every(Number.isFinite) &&
          target.rect.width > 0 &&
          target.rect.height > 0
      )
      .map((target) => ({ target, gap: rectDistance(point, target.rect) }))
      .filter((item) => item.gap <= radius + (item.target.id === this.current?.id ? retention : 0))
      .sort(
        (a, b) =>
          a.gap - b.gap || a.target.rect.width * a.target.rect.height - b.target.rect.width * b.target.rect.height
      )
    let next: HandTarget | null = ranked[0]?.target ?? null
    const previous = ranked.find((item) => item.target.id === this.current?.id)
    // A directly pointed inner control overrides its container. Otherwise retain a nearby target.
    if (
      previous &&
      !(
        ranked[0].gap === 0 &&
        next?.id !== previous.target.id &&
        (previous.gap > 8 || contains(previous.target.rect, next!.rect))
      )
    )
      next = previous.target
    else if (
      ranked[1] &&
      ranked[1].gap - ranked[0].gap < 10 &&
      !contains(ranked[1].target.rect, ranked[0].target.rect)
    ) {
      result.ambiguous = ranked
        .filter((item) => item.gap - ranked[0].gap < 10)
        .slice(0, 4)
        .map((item) => item.target)
      next = null
    }
    if (!next) {
      if (result.ambiguous.length) {
        const key = result.ambiguous
          .map((target) => target.id)
          .sort()
          .join('|')
        if (key !== this.ambiguity || !this.anchor || distance(this.anchor, point) > 12 || !dwellEnabled) {
          this.anchor = point
          this.since = now
        }
        this.current = null
        this.ambiguity = key
        result.settled = dwellEnabled && now - this.since >= 350
      } else this.reset()
      return result
    }
    this.ambiguity = ''
    const shifted = this.current && distance(targetPoint(this.current), targetPoint(next)) > 8
    if (this.current?.id !== next.id || shifted || !this.anchor || distance(this.anchor, point) > 12 || !dwellEnabled) {
      this.anchor = point
      this.since = now
    }
    this.current = next
    result.target = next
    // Pull toward the closest usable edge, not the center of a potentially huge header.
    // Spatial falloff starts at zero at the acquisition boundary: no animation lag or snap.
    const gap = rectDistance(point, next.rect)
    const pull = magnetic ? this.strength * Math.max(0, 1 - gap / radius) : 0
    const insetX = Math.min(8, next.rect.width / 2)
    const insetY = Math.min(8, next.rect.height / 2)
    const x = Math.max(next.rect.x + insetX, Math.min(point.x, next.rect.x + next.rect.width - insetX))
    const y = Math.max(next.rect.y + insetY, Math.min(point.y, next.rect.y + next.rect.height - insetY))
    result.point = { x: point.x + (x - point.x) * pull, y: point.y + (y - point.y) * pull }
    result.settled = dwellEnabled && now - this.since >= 350
    if (!dwellEnabled || this.blocked || now < this.cooldownUntil || next.activation !== 'dwell') return result
    // A quiet settling interval avoids flashing timers as the hand passes through targets.
    result.progress = Math.min(1, Math.max(0, (now - this.since - 180) / this.dwellMs))
    if (result.progress === 1) {
      result.activate = next
      this.block(next, now)
    }
    return result
  }
}

/** Relative palm movement, followed by a deliberate pause, for one-hand move/resize/pan. */
export class HoldToPlace {
  private anchor: HandPoint
  private since: number
  private lastTime: number
  private moved = false
  constructor(
    readonly origin: HandPoint,
    now: number,
    readonly holdMs = 900
  ) {
    this.anchor = origin
    this.since = this.lastTime = now
  }
  update(point: HandPoint, now: number): { delta: HandPoint; progress: number; done: boolean } {
    const delta = { x: point.x - this.origin.x, y: point.y - this.origin.y }
    this.moved ||= distance(point, this.origin) > 12
    if (now - this.lastTime > 150 || now <= this.lastTime || distance(point, this.anchor) > 10) {
      this.anchor = point
      this.since = now
    }
    this.lastTime = now
    const progress = this.moved ? Math.min(1, Math.max(0, (now - this.since) / this.holdMs)) : 0
    return { delta, progress, done: progress === 1 }
  }
}
