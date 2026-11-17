import { describe, expect, it } from 'vitest'
import { HandSnapGesture } from './snap.js'
import type { HandFrame, HandObservation } from './gesture.js'
import { snapHand } from './snap.test-fixture.js'

function fixture(dt = 1000 / 30, transform = (hands: HandObservation[]) => hands) {
  const gesture = new HandSnapGesture(1)
  let time = 1000
  let triggers = 0
  const feed = (hands: HandObservation[], elapsed = dt) => {
    time += elapsed
    const result = gesture.step({ gen: 1, wallTs: time, ts: time, aspect: 4 / 3, hands: transform(hands) }, time)
    if (result.triggered) triggers++
    return result
  }
  const step = (pose: Parameters<typeof snapHand>[0], elapsed = dt) => feed([snapHand(pose)], elapsed)
  const release = () => {
    for (let i = 0; i < 4; i++) step('release')
  }
  const snap = () => {
    step('contact')
    release()
  }
  return { gesture, feed, step, release, snap, triggers: () => triggers, time: () => time }
}

describe('middle finger snap', () => {
  it.each([15, 24, 30, 60])(
    'recognizes a fast snap with one contact frame at %i fps and no open-hand ritual',
    (fps) => {
      const f = fixture(1000 / fps)
      expect(f.step('contact')).toMatchObject({ active: true, triggered: false })
      expect(f.step('release').triggered).toBe(false)
      f.release()
      expect(f.triggers()).toBe(1)
      for (let i = 0; i < fps; i++) f.step('release')
      f.snap()
      expect(f.triggers()).toBe(2)
    }
  )

  it('allows a prepared contact to be held without toggling until the inward flick', () => {
    const f = fixture()
    for (let i = 0; i < 90; i++) f.step('contact')
    expect(f.triggers()).toBe(0)
    f.release()
    expect(f.triggers()).toBe(1)
  })

  it('debounces follow-through, contact bounce and repeats during cooldown', () => {
    const f = fixture()
    f.snap()
    f.snap()
    f.snap()
    expect(f.triggers()).toBe(1)
    for (let i = 0; i < 30; i++) f.step('release')
    f.snap()
    expect(f.triggers()).toBe(2)
  })

  it.each(['opening', 'thumb-only', 'flicker', 'slow-release', 'index-pinch'] as const)(
    'does not mistake %s for a snap',
    (reason) => {
      const f = fixture()
      f.step('contact')
      if (reason === 'opening') for (let i = 0; i < 4; i++) f.step('open')
      if (reason === 'thumb-only') {
        const hand = snapHand('contact')
        hand.landmarks[4] = { x: 0.2, y: 0.48 }
        for (let i = 0; i < 4; i++) f.feed([hand])
      }
      if (reason === 'flicker') {
        f.step('release')
        f.step('contact')
        f.release()
      }
      if (reason === 'slow-release') {
        for (let i = 0; i < 8; i++) f.step('separating')
        f.release()
      }
      if (reason === 'index-pinch') {
        for (let i = 0; i < 4; i++) f.step('pinch')
        f.release()
      }
      expect(f.triggers()).toBe(0)
    }
  )

  it('tracks either snapping hand beside a passive hand even when detection order changes', () => {
    for (const dx of [0, 0.3]) {
      let reverse = false
      const f = fixture(40, (hands) => {
        const active = hands[0]
        active.landmarks = active.landmarks.map((p) => ({ x: p.x + dx, y: p.y }))
        const pair = [active, snapHand('open', dx ? 0 : 0.3)]
        reverse = !reverse
        return reverse ? pair.reverse() : pair
      })
      f.snap()
      expect(f.triggers()).toBe(1)
    }
  })

  it('yields immediately to an index pinch on the other hand for two-hand navigation', () => {
    const f = fixture()
    f.step('contact')
    expect(f.feed([snapHand('release'), snapHand('pinch', 0.3)]).active).toBe(false)
    f.release()
    expect(f.triggers()).toBe(0)
  })

  it.each(['loss', 'jump', 'malformed', 'gap', 'abort'] as const)('cannot complete a snap across %s', (reason) => {
    const f = fixture()
    f.step('contact')
    if (reason === 'abort') f.gesture.abort()
    else if (reason === 'gap') f.step('release', 151)
    else if (reason === 'loss') f.feed([])
    else {
      const hand = snapHand('release', reason === 'jump' ? 0.3 : 0)
      if (reason === 'malformed') hand.landmarks[12].x = NaN
      f.feed([hand])
    }
    f.release()
    expect(f.triggers()).toBe(0)
  })

  it('recognizes rotated and mirrored snaps while the hand moves through the camera', () => {
    for (const angle of [0, Math.PI / 2, Math.PI])
      for (const mirror of [-1, 1]) {
        let dx = 0
        const f = fixture(40, (hands) =>
          hands.map((h) => ({
            handedness: 'Right',
            landmarks: h.landmarks.map((p) => {
              const x = (p.x - 0.5) * mirror,
                y = (p.y - 0.5) / (4 / 3)
              return {
                x: 0.5 + dx + x * Math.cos(angle) - y * Math.sin(angle),
                y: 0.5 + (x * Math.sin(angle) + y * Math.cos(angle)) * (4 / 3)
              }
            })
          }))
        )
        for (let i = 0; i < 6; i++) {
          f.step('contact')
          dx += 0.025
        }
        f.release()
        expect(f.triggers()).toBe(1)
      }
  })

  it('rejects stale, reordered, future and wrong-generation frames', () => {
    const gesture = new HandSnapGesture(1)
    const frame: HandFrame = { gen: 1, wallTs: 1000, ts: 1, aspect: 4 / 3, hands: [snapHand()] }
    for (const [sample, now] of [
      [frame, 1151],
      [frame, 999],
      [{ ...frame, gen: 2 }, 1000]
    ] as const)
      expect(gesture.step(sample, now).accepted).toBe(false)
    expect(gesture.step(frame, 1000).accepted).toBe(true)
    expect(gesture.step(frame, 1000).accepted).toBe(false)
  })
})
