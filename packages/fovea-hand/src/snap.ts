import { HAND_FRAME_MAX_AGE_MS, type HandFrame, type HandPoint, pinchMetric } from './gesture.js'
import { fingerMetrics, metricDistance } from './fingerPose.js'

/** A visual snap is thumb/middle contact followed by the middle finger flicking
 * toward the palm. Contact may already be held when the hand enters the camera.
 * Reserve pointer input while active; the host decides what a snap does. */
export class HandSnapGesture {
  private phase: 'idle' | 'contact' | 'release' | 'cooldown' = 'idle'
  private lastTs = -Infinity
  private lastContact = 0
  private contactReach = 0
  private releasedAt = 0
  private cooldownUntil = 0
  private wrist: HandPoint | null = null

  constructor(readonly gen: number) {}

  abort(): void {
    this.phase = 'idle'
    this.wrist = null
  }

  step(frame: HandFrame, now: number): { accepted: boolean; active: boolean; triggered: boolean } {
    const wasActive = this.phase === 'contact' || this.phase === 'release'
    const result = (active = wasActive, triggered = false) => ({ accepted: true, active, triggered })
    if (
      frame.gen !== this.gen ||
      !Number.isFinite(now) ||
      !Number.isFinite(frame.wallTs) ||
      !Number.isFinite(frame.ts) ||
      frame.wallTs > now ||
      now - frame.wallTs > HAND_FRAME_MAX_AGE_MS ||
      frame.wallTs <= this.lastTs
    )
      return { accepted: false, active: wasActive, triggered: false }

    const gap = frame.wallTs - this.lastTs
    this.lastTs = frame.wallTs
    if (
      !frame.hands.length ||
      frame.hands.length > 2 ||
      frame.hands.some((hand) => !fingerMetrics(hand, frame.aspect)) ||
      (Number.isFinite(gap) && gap > HAND_FRAME_MAX_AGE_MS)
    ) {
      this.abort()
      return result(false)
    }
    const hands = frame.hands.map((hand) => {
      const points = hand.landmarks
      const palm = metricDistance(points[0], points[9], frame.aspect)
      const middle = metricDistance(points[4], points[12], frame.aspect) / palm
      const index = pinchMetric(points, frame.aspect)
      return {
        wrist: points[0],
        middle,
        reach: metricDistance(points[0], points[12], frame.aspect) / palm,
        // A relaxed index may be nearby. Only an actual index pinch takes priority.
        indexPinch: index < 0.35 && index <= middle + 0.08,
        contact: middle < 0.4 && index > middle + 0.08
      }
    })
    if (hands.some((hand) => hand.indexPinch)) {
      this.abort()
      return result(false)
    }
    // Follow the same wrist even if MediaPipe changes detection order or labels.
    const wrist = this.wrist
    const hand = wrist
      ? hands.reduce((nearest, next) =>
          metricDistance(next.wrist, wrist, frame.aspect) < metricDistance(nearest.wrist, wrist, frame.aspect)
            ? next
            : nearest
        )
      : hands.find((candidate) => candidate.contact)
    if (!hand || (wrist && metricDistance(hand.wrist, wrist, frame.aspect) > 0.12)) {
      this.abort()
      return result(false)
    }
    this.wrist = hand.wrist
    const separated = hand.middle > 0.65
    if (this.phase === 'cooldown') {
      if (separated && frame.wallTs >= this.cooldownUntil) this.abort()
      return result(false)
    }
    if (this.phase === 'idle') {
      if (!hand.contact || frame.wallTs < this.cooldownUntil) return result(false)
      this.phase = 'contact'
      this.lastContact = frame.wallTs
      this.contactReach = hand.reach
      return result(true)
    }
    if (this.phase === 'contact') {
      if (hand.contact) {
        this.lastContact = frame.wallTs
        this.contactReach = hand.reach
      } else if (frame.wallTs - this.lastContact > 180) this.phase = 'cooldown'
      else if (separated) {
        // A fast snap can expose only ONE contact frame at webcam frame rates.
        // The inward flick and a second separated sample distinguish it from
        // contact noise, moving only the thumb, or simply opening the hand.
        if (this.contactReach - hand.reach >= 0.18) {
          this.phase = 'release'
          this.releasedAt = frame.wallTs
        } else this.phase = 'cooldown'
      }
      return result(this.phase === 'contact' || this.phase === 'release')
    }
    if (!separated || this.contactReach - hand.reach < 0.18) {
      this.phase = 'cooldown'
      return result(true)
    }
    if (frame.wallTs - this.releasedAt < 20) return result(true)
    this.phase = 'cooldown'
    this.cooldownUntil = frame.wallTs + 800
    return result(true, true)
  }
}
