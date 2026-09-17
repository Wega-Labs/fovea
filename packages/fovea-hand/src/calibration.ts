import type { HandPoint, HandRegion } from './gesture.js'

/** Captures only normalized landmark positions, never camera pixels. */
export class HandReachCalibration {
  private points: HandPoint[] = []
  constructor(readonly startedAt: number) {}
  add(point: HandPoint): void {
    if (
      Number.isFinite(point.x) &&
      Number.isFinite(point.y) &&
      point.x >= 0 &&
      point.x <= 1 &&
      point.y >= 0 &&
      point.y <= 1
    )
      this.points.push(point)
  }
  finish(): HandRegion | null {
    if (this.points.length < 30) return null
    const xs = this.points.map((p) => p.x).sort((a, b) => a - b),
      ys = this.points.map((p) => p.y).sort((a, b) => a - b)
    const low = Math.floor(xs.length * 0.05),
      high = Math.ceil(xs.length * 0.95) - 1
    if (xs[high] - xs[low] < 0.15 || ys[high] - ys[low] < 0.15) return null
    return {
      left: Math.max(0, xs[low] - 0.04),
      right: Math.min(1, xs[high] + 0.04),
      top: Math.max(0, ys[low] - 0.04),
      bottom: Math.min(1, ys[high] + 0.04)
    }
  }
}
