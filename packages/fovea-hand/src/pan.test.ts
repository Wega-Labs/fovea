import { describe, expect, it } from 'vitest'
import { type HandFrame, type HandObservation } from './gesture.js'
import { HandPanGesture, isOpenHandPose, isTwoFingerPanPose, type HandPanIntent } from './pan.js'

function hand(dx = 0, dy = 0, pose: 'pan' | 'open' | 'pinch' | 'folded' = 'pan'): HandObservation {
  const landmarks = Array.from({ length: 21 }, () => ({ x: 0.5, y: 0.6 }))
  landmarks[0] = { x: 0.5, y: 0.75 }
  landmarks[4] = { x: 0.25, y: 0.5 }
  for (const [finger, x] of [
    [5, 0.42],
    [9, 0.49],
    [13, 0.56],
    [17, 0.63]
  ]) {
    const extended = pose === 'open' || (pose !== 'folded' && finger <= 9)
    landmarks[finger] = { x, y: 0.55 }
    landmarks[finger + 1] = { x, y: 0.44 }
    landmarks[finger + 2] = { x, y: extended ? 0.35 : 0.48 }
    landmarks[finger + 3] = { x, y: extended ? 0.27 : 0.56 }
  }
  if (pose === 'pinch') landmarks[4] = { ...landmarks[8], x: landmarks[8].x + 0.02 }
  return { handedness: 'Left', landmarks: landmarks.map((point) => ({ x: point.x + dx, y: point.y + dy })) }
}

function fixture() {
  const machine = new HandPanGesture(7)
  let time = 1000
  const events: HandPanIntent[] = []
  const step = (hands: HandObservation[], dt = 40) => {
    time += dt
    const frame: HandFrame = { gen: 7, wallTs: time, ts: time, aspect: 4 / 3, hands }
    const result = machine.step(frame, time)
    events.push(...result.intents)
    return result
  }
  const start = () => {
    step([hand()])
    step([hand(-0.01)])
    step([hand(-0.02)])
    const result = step([hand(-0.04)])
    expect(machine.phase).toBe('panning')
    expect(result.intents).toEqual([{ type: 'pan', phase: 'start', delta: { x: 0, y: 0 } }])
  }
  return { machine, step, events, start }
}

describe('two-finger pose', () => {
  it('resumes pointing only with four relaxed extended fingers and no pinch', () => {
    expect(isOpenHandPose(hand(0, 0, 'open'), 4 / 3)).toBe(true)
    for (const pose of ['folded', 'pan', 'pinch'] as const) expect(isOpenHandPose(hand(0, 0, pose), 4 / 3)).toBe(false)
    const relaxed = hand(0, 0, 'open')
    for (const tip of [8, 12, 16, 20]) {
      relaxed.landmarks[tip].x += 0.06
      relaxed.landmarks[tip].y = 0.37
    }
    expect(isOpenHandPose(relaxed, 4 / 3)).toBe(true)
    const pinched = hand(0, 0, 'open')
    pinched.landmarks[4] = { ...pinched.landmarks[8], x: pinched.landmarks[8].x + 0.02 }
    expect(isOpenHandPose(pinched, 4 / 3)).toBe(false)
  })
  it('recognizes rotated open hands and rejects malformed observations', () => {
    const rotated = hand(0, 0, 'open')
    rotated.landmarks = rotated.landmarks.map((point) => ({
      x: 0.5 - (point.y - 0.5) / (4 / 3),
      y: 0.5 + (point.x - 0.5) * (4 / 3)
    }))
    expect(isOpenHandPose(rotated, 4 / 3)).toBe(true)
    for (const aspect of [0, -1, NaN, Infinity]) expect(isOpenHandPose(hand(0, 0, 'open'), aspect)).toBe(false)
    expect(isOpenHandPose({ ...hand(), landmarks: hand().landmarks.slice(0, 20) }, 4 / 3)).toBe(false)
    const invalid = hand(0, 0, 'open')
    invalid.landmarks[8].x = NaN
    expect(isOpenHandPose(invalid, 4 / 3)).toBe(false)
    expect(
      isOpenHandPose({ ...hand(), landmarks: Array.from({ length: 21 }, () => ({ x: 0.5, y: 0.5 })) }, 4 / 3)
    ).toBe(false)
  })
  it('recognizes the deliberate pose but rejects open hands, pinches and folded fingers', () => {
    expect(isTwoFingerPanPose(hand(), 4 / 3)).toBe(true)
    for (const pose of ['open', 'pinch', 'folded'] as const)
      expect(isTwoFingerPanPose(hand(0, 0, pose), 4 / 3)).toBe(false)
    const indexOnly = hand()
    for (let i = 10; i <= 12; i++) indexOnly.landmarks[i] = { x: 0.49, y: i === 10 ? 0.44 : 0.56 }
    expect(isTwoFingerPanPose(indexOnly, 4 / 3)).toBe(false)
  })
  it('works when the hand rotates or mirrors in the camera plane', () => {
    for (const angle of [0, Math.PI / 3, Math.PI / 2, Math.PI]) {
      for (const mirror of [-1, 1]) {
        const observation = hand()
        observation.landmarks = observation.landmarks.map((point) => {
          const x = (point.x - 0.5) * mirror,
            y = (point.y - 0.5) / (4 / 3)
          return {
            x: 0.5 + x * Math.cos(angle) - y * Math.sin(angle),
            y: 0.5 + (x * Math.sin(angle) + y * Math.cos(angle)) * (4 / 3)
          }
        })
        expect(isTwoFingerPanPose(observation, 4 / 3)).toBe(true)
      }
    }
  })
  it('rejects incomplete, nonfinite, degenerate and edge-on observations', () => {
    expect(isTwoFingerPanPose({ ...hand(), landmarks: hand().landmarks.slice(0, 20) }, 4 / 3)).toBe(false)
    for (const aspect of [0, -1, NaN, Infinity]) expect(isTwoFingerPanPose(hand(), aspect)).toBe(false)
    const observation = hand()
    observation.landmarks[8].x = NaN
    expect(isTwoFingerPanPose(observation, 4 / 3)).toBe(false)
    expect(
      isTwoFingerPanPose({ ...hand(), landmarks: Array.from({ length: 21 }, () => ({ x: 0.5, y: 0.5 })) }, 4 / 3)
    ).toBe(false)
  })
})

describe('continuous hand panning', () => {
  it('reserves the pose immediately but requires stable elapsed time and deliberate travel before starting', () => {
    const f = fixture()
    f.step([hand()])
    expect(f.machine.phase).toBe('arming')
    for (let i = 0; i < 8; i++) f.step([hand(i % 2 ? 0.001 : -0.001)])
    expect(f.machine.phase).toBe('arming')
    expect(f.events).toEqual([])
    f.step([hand(-0.04)])
    expect(f.machine.phase).toBe('panning')
    expect(f.events).toEqual([{ type: 'pan', phase: 'start', delta: { x: 0, y: 0 } }])

    const quick = fixture()
    quick.step([hand()])
    quick.step([hand(-0.05)], 30)
    quick.step([hand(-0.08)], 30)
    expect(quick.machine.phase).toBe('arming')
    expect(quick.events).toEqual([])
  })
  it('mirrors offset direction and preserves vertical direction', () => {
    const f = fixture()
    f.start()
    const moves = f.step([hand(-0.06, 0.02)]).intents
    expect(moves).toHaveLength(1)
    expect(moves[0].phase).toBe('move')
    expect(moves[0].delta.x).toBeGreaterThan(0)
    expect(moves[0].delta.x).toBeLessThan(0.04)
    expect(moves[0].delta.y).toBeGreaterThan(0)
    f.step([hand(0.03, -0.03)])
    const next = f.step([hand(0.03, -0.03)]).intents[0]
    expect(next.delta.x).toBeLessThan(0)
    expect(next.delta.y).toBeLessThan(0)
  })
  it('keeps traveling at a held offset and stops immediately at neutral without filter drift', () => {
    const f = fixture()
    f.start()
    for (let i = 0; i < 30; i++) {
      const move = f.step([hand(-0.08)]).intents[0]
      expect(move.delta.x).toBeGreaterThan(0.001)
    }
    for (let i = 0; i < 10; i++) expect(f.step([hand()]).intents[0].delta).toEqual({ x: 0, y: 0 })
    expect(f.machine.phase).toBe('panning')
  })

  it('covers the same distance at different camera rates once settled', () => {
    const travel = (dt: number) => {
      const f = fixture()
      f.start()
      for (let i = 0; i < 25; i++) f.step([hand(-0.09)])
      let distance = 0
      for (let t = 0; t < 1000; t += dt) distance += f.step([hand(-0.09)], dt).intents[0].delta.x
      return distance
    }
    expect(travel(20)).toBeCloseTo(travel(50), 5)
  })
  it('stops on the first broken-pose frame and lets the hand reposition without canvas movement', () => {
    const f = fixture()
    f.start()
    f.step([hand(-0.06)])
    expect(f.step([hand(-0.08, 0, 'folded')]).intents).toEqual([{ type: 'pan', phase: 'end', delta: { x: 0, y: 0 } }])
    expect(f.machine.phase).toBe('idle')
    expect(f.step([hand(0.1, 0, 'folded')]).intents).toEqual([])
    expect(f.step([hand(0.1)]).intents).toEqual([])
    expect(f.machine.phase).toBe('arming')
    f.step([hand(0.08)])
    f.step([hand(0.06)])
    expect(f.step([hand(0.04)]).intents).toEqual([{ type: 'pan', phase: 'start', delta: { x: 0, y: 0 } }])
  })
  it('does not start from open hands, one-hand pinches, or two-hand pinches', () => {
    const f = fixture()
    for (const hands of [
      [hand(0, 0, 'open')],
      [hand(0, 0, 'pinch')],
      [hand(0, 0, 'pinch'), hand(0.2, 0, 'pinch')],
      [hand(), hand(0.2)]
    ]) {
      for (let i = 0; i < 8; i++) f.step(hands)
      expect(f.machine.phase).toBe('idle')
      expect(f.events).toEqual([])
    }
  })
  it('ends immediately on transition to a pinch without movement on that frame', () => {
    const f = fixture()
    f.start()
    expect(f.step([hand(-0.06, 0, 'pinch')]).intents).toEqual([{ type: 'pan', phase: 'end', delta: { x: 0, y: 0 } }])
  })
  it('cancels once on hand loss, a second hand, and a tracking discontinuity', () => {
    for (const hands of [[], [hand(), hand(0.2)], [hand(0.5)]]) {
      const f = fixture()
      f.start()
      expect(f.step(hands).intents).toEqual([{ type: 'pan', phase: 'cancel', delta: { x: 0, y: 0 } }])
      expect(f.machine.phase).toBe('idle')
      expect(f.machine.abort()).toEqual([])
    }
  })
  it('does not confuse handedness label flips with tracking discontinuity', () => {
    const f = fixture()
    f.start()
    expect(f.step([{ ...hand(-0.06), handedness: 'Right' }]).intents[0].phase).toBe('move')
  })
  it.each([151, 250, 399])('rebases after a fresh %ims gap without replaying movement or ending the stroke', (gap) => {
    const f = fixture()
    f.start()
    expect(f.step([hand(-0.15)], gap).intents).toEqual([])
    expect(f.machine.phase).toBe('panning')
    const move = f.step([hand(-0.18)]).intents[0]
    expect(move.phase).toBe('move')
    expect(move.delta.x).toBeGreaterThan(0)
    expect(move.delta.x).toBeLessThan(0.01)
    expect(f.events.filter((event) => event.phase === 'start')).toHaveLength(1)
  })
  it.each([400, 401])('cancels once after a %ims tracking gap and rearms on a later fresh pose', (gap) => {
    const f = fixture()
    f.start()
    expect(f.step([hand(-0.06)], gap).intents).toEqual([{ type: 'pan', phase: 'cancel', delta: { x: 0, y: 0 } }])
    expect(f.machine.phase).toBe('idle')
    expect(f.step([hand(-0.07)]).intents).toEqual([])
    expect(f.machine.phase).toBe('arming')
  })
  it('continues a stroke through relaxed pose boundary jitter without relaxing acquisition', () => {
    const relaxed = (dx: number): HandObservation => {
      const observation = hand(dx)
      observation.landmarks[8].x += 0.07
      observation.landmarks[8].y = 0.36
      return observation
    }
    expect(isTwoFingerPanPose(relaxed(0), 4 / 3)).toBe(false)
    const f = fixture()
    for (let i = 0; i < 4; i++) f.step([relaxed(-i * 0.01)])
    expect(f.machine.phase).toBe('idle')
    f.start()
    for (let i = 1; i <= 4; i++) {
      const event = f.step([i % 2 ? relaxed(-0.04 - i * 0.01) : hand(-0.04 - i * 0.01)]).intents[0]
      expect(event.phase).toBe('move')
      expect(event.delta.x).toBeGreaterThan(0)
      expect(f.machine.phase).toBe('panning')
    }
  })
  it('pauses one ambiguous sample and rebases recovery without a catchup jump', () => {
    const f = fixture()
    f.start()
    const ambiguous = hand(-0.06)
    ambiguous.landmarks[8].x += 0.16
    ambiguous.landmarks[8].y = 0.38
    expect(f.step([ambiguous]).intents).toEqual([])
    expect(f.machine.phase).toBe('panning')
    expect(f.step([hand(-0.12)]).intents).toEqual([])
    expect(f.machine.phase).toBe('panning')
    const move = f.step([hand(-0.15)]).intents[0]
    expect(move.phase).toBe('move')
    expect(move.delta.x).toBeGreaterThan(0)
    expect(move.delta.x).toBeLessThan(0.01)
    expect(f.events.filter((event) => event.phase === 'start')).toHaveLength(1)
  })
  it('ends after at most 80ms of uncertain pose while explicit folding ends on the first frame', () => {
    const f = fixture()
    f.start()
    const ambiguous = hand(-0.06)
    ambiguous.landmarks[8].x += 0.16
    ambiguous.landmarks[8].y = 0.38
    expect(f.step([ambiguous]).intents).toEqual([])
    expect(f.step([ambiguous], 79).intents).toEqual([])
    expect(f.step([ambiguous], 1).intents).toEqual([{ type: 'pan', phase: 'end', delta: { x: 0, y: 0 } }])
    expect(f.machine.phase).toBe('idle')
    f.start()
    expect(f.step([hand(-0.06, 0, 'folded')], 1).intents).toEqual([
      { type: 'pan', phase: 'end', delta: { x: 0, y: 0 } }
    ])
  })
  it('does not extend uncertain-pose grace when the next valid sample arrives after 80ms', () => {
    const f = fixture()
    f.start()
    const ambiguous = hand(-0.06)
    ambiguous.landmarks[8].x += 0.16
    ambiguous.landmarks[8].y = 0.38
    f.step([ambiguous])
    expect(f.step([hand(-0.12)], 81).intents).toEqual([{ type: 'pan', phase: 'end', delta: { x: 0, y: 0 } }])
    expect(f.machine.phase).toBe('idle')
  })
  it('rejects stale, future, wrong-generation and reordered frames without refreshing state', () => {
    const machine = new HandPanGesture(7)
    const frame: HandFrame = { gen: 7, wallTs: 1000, ts: 1, aspect: 4 / 3, hands: [hand()] }
    expect(machine.step({ ...frame, gen: 6 }, 1000).accepted).toBe(false)
    expect(machine.step(frame, 1151).accepted).toBe(false)
    expect(machine.step(frame, 999).accepted).toBe(false)
    expect(machine.step(frame, NaN).accepted).toBe(false)
    expect(machine.phase).toBe('idle')
    expect(machine.step(frame, 1100).accepted).toBe(true)
    expect(machine.step(frame, 1100).accepted).toBe(false)
  })
  it('keeps relative movement available past the pointer active region', () => {
    const f = fixture()
    f.start()
    for (let i = 1; i < 12; i++) {
      const event = f.step([hand(-0.04 - i * 0.04)]).intents[0]
      expect(event.phase).toBe('move')
      expect(event.delta.x).toBeGreaterThan(0.015)
    }
  })
})
