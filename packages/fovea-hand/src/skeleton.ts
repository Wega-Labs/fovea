import { DrawingUtils, HandLandmarker } from '@mediapipe/tasks-vision'
import { pinchMetric, DEFAULT_HAND_REGION, type HandRegion, type HandFrame } from './gesture.js'

/** Draw only landmarks, using MediaPipe's own hand connections. Camera pixels stay in the worker. */
export function drawHandSkeleton(
  canvas: HTMLCanvasElement,
  frame: HandFrame,
  region: HandRegion = DEFAULT_HAND_REGION
): number {
  const hands = frame.hands
    .slice(0, 2)
    .filter(
      (hand) =>
        hand.landmarks.length === 21 &&
        hand.landmarks.every((point) => Number.isFinite(point.x) && Number.isFinite(point.y))
    )
  const context = canvas.getContext('2d')
  if (!context) return hands.length
  const height = Math.round(canvas.width / frame.aspect)
  if (canvas.height !== height) canvas.height = height
  context.clearRect(0, 0, canvas.width, canvas.height)
  context.strokeStyle = '#ffffff30'
  context.lineWidth = 2
  context.setLineDash([8, 8])
  context.strokeRect(
    canvas.width * (1 - region.right),
    canvas.height * region.top,
    canvas.width * (region.right - region.left),
    canvas.height * (region.bottom - region.top)
  )
  context.setLineDash([])
  const drawing = new DrawingUtils(context)
  try {
    hands.forEach((hand, index) => {
      const points = hand.landmarks.map(({ x, y }) => ({ x: 1 - x, y, z: 0, visibility: 1 }))
      const color = index === 0 ? '#80afff' : '#66dec2'
      drawing.drawConnectors(points, HandLandmarker.HAND_CONNECTIONS, { color, lineWidth: 4 })
      drawing.drawLandmarks(points, { color, fillColor: '#10151e', lineWidth: 2, radius: 4 })
      const closed = pinchMetric(hand.landmarks, frame.aspect) < 0.35
      drawing.drawConnectors(points, [{ start: 4, end: 8 }], {
        color: closed ? '#ffd083' : '#ffffff60',
        lineWidth: closed ? 6 : 2
      })
      drawing.drawLandmarks([points[4], points[8]], {
        color: closed ? '#ffd083' : '#ffffff',
        fillColor: closed ? '#ffd083' : '#10151e',
        lineWidth: 3,
        radius: 6
      })
    })
  } finally {
    drawing.close()
  }
  return hands.length
}
