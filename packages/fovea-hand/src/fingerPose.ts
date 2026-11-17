import type { HandObservation, HandPoint } from './gesture.js'

export const metricDistance = (a: HandPoint, b: HandPoint, aspect: number): number =>
  Math.hypot(a.x - b.x, (a.y - b.y) / aspect)

export interface FingerMetrics {
  straightness: number
  reach: number
}

export function fingerMetrics(hand: HandObservation, aspect: number): FingerMetrics[] | null {
  if (
    !Number.isFinite(aspect) ||
    aspect <= 0 ||
    hand.landmarks.length !== 21 ||
    !hand.landmarks.every((point) => Number.isFinite(point.x) && Number.isFinite(point.y))
  )
    return null
  const points = hand.landmarks
  const palmSize = metricDistance(points[0], points[9], aspect)
  if (palmSize < 0.02) return null
  return [5, 9, 13, 17].map((mcp) => {
    const base = points[mcp],
      joint = points[mcp + 1],
      tip = points[mcp + 3]
    const a = { x: joint.x - base.x, y: (joint.y - base.y) / aspect }
    const b = { x: tip.x - joint.x, y: (tip.y - joint.y) / aspect }
    const lengths = Math.hypot(a.x, a.y) * Math.hypot(b.x, b.y)
    return {
      straightness: lengths > 0.00001 ? (a.x * b.x + a.y * b.y) / lengths : 0,
      reach: (metricDistance(tip, points[0], aspect) - metricDistance(joint, points[0], aspect)) / palmSize
    }
  })
}
