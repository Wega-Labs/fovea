import { describe, expect, it } from 'vitest'
import { HandGesture, type HandIntent, type HandObservation } from './gesture.js'

function hand(pose: 'open' | 'curl' | 'fist' | 'half' = 'open', dx = 0): HandObservation {
  const landmarks = Array.from({ length: 21 }, () => ({ x: 0.5, y: 0.6 }))
  landmarks[0] = { x: 0.5, y: 0.75 }
  landmarks[4] = { x: 0.25, y: 0.5 }
  for (const [mcp, x] of [
    [5, 0.42],
    [9, 0.49],
    [13, 0.56],
    [17, 0.63]
  ]) {
    const folded = pose === 'fist' || (pose === 'curl' && mcp === 5)
    landmarks[mcp] = { x, y: 0.55 }
    landmarks[mcp + 1] = { x, y: 0.44 }
    landmarks[mcp + 2] = { x, y: folded ? 0.48 : 0.35 }
    landmarks[mcp + 3] = pose === 'half' && mcp === 5 ? { x: x + 0.1, y: 0.4 } : { x, y: folded ? 0.56 : 0.27 }
  }
  return { handedness: 'Left', landmarks: landmarks.map((point) => ({ x: point.x + dx, y: point.y })) }
}

function fixture(transform = (h: HandObservation) => h) {
  const gesture = new HandGesture(1, undefined, 'curl')
  const events: HandIntent[] = []
  const target = { x: 530, y: 220 }
  let time = 1000
  const step = (hands: HandObservation[], dt = 40) => {
    time += dt
    const result = gesture.step(
      { gen: 1, wallTs: time, ts: time, aspect: 4 / 3, hands: hands.map(transform) },
      time,
      1601,
      1001,
      target
    )
    events.push(...result.intents)
    return result.intents
  }
  const curl = () => {
    step([hand()])
    for (let i = 0; i < 3; i++) step([hand('curl')])
    expect(gesture.phase).toBe('curling')
  }
  const release = (dx = 0) => {
    for (let i = 0; i < 3; i++) step([hand('open', dx)])
  }
  const clicks = () => events.filter((event) => event.type === 'click')
  return { gesture, events, target, step, curl, release, clicks }
}

describe('index-finger curl clicks', () => {
  it('accepts index-only pointing while the other fingers rest folded throughout the click', () => {
    const f = fixture((h) => {
      const folded = hand('fist')
      for (const mcp of [9, 13, 17]) {
        h.landmarks[mcp + 2] = folded.landmarks[mcp + 2]
        h.landmarks[mcp + 3] = folded.landmarks[mcp + 3]
      }
      return h
    })
    f.curl()
    f.release()
    expect(f.clicks()).toEqual([{ type: 'click', ...f.target }])
  })

  it('holds the pre-curl target through finger and palm travel and clicks once on straightening', () => {
    const f = fixture()
    f.curl()
    f.step([hand('curl', 0.06)])
    expect(f.clicks()).toEqual([])
    f.release(0.06)
    expect(f.clicks()).toEqual([{ type: 'click', ...f.target }])
    for (let i = 0; i < 20; i++) f.step([hand('open', 0.06)])
    expect(f.clicks()).toHaveLength(1)
    expect(f.events.some((event) => event.type === 'down' || event.type === 'up')).toBe(false)
    f.curl()
    f.release()
    expect(f.clicks()).toHaveLength(2)
  })

  it.each(['mirror', 'rotate'] as const)('recognizes a %s hand in the camera plane', (kind) => {
    const f = fixture((h) => ({
      ...h,
      landmarks: h.landmarks.map(({ x, y }) =>
        kind === 'mirror' ? { x: 1 - x, y } : { x: 0.5 - (y - 0.5) / (4 / 3), y: 0.5 + (x - 0.5) * (4 / 3) }
      )
    }))
    f.curl()
    f.release()
    expect(f.clicks()).toHaveLength(1)
  })

  it('ignores a bent hand on entry, a shallow bend, and one noisy folded frame', () => {
    const f = fixture()
    for (let i = 0; i < 5; i++) f.step([hand('curl')])
    f.release()
    f.step([hand('half')])
    f.release()
    f.step([hand('curl')])
    f.release()
    expect(f.clicks()).toEqual([])
  })

  it.each(['loss', 'gap', 'departure', 'timeout', 'fist', 'second hand', 'abort'] as const)(
    'cancels on %s without a delayed click',
    (reason) => {
      const f = fixture()
      f.curl()
      if (reason === 'loss') f.step([])
      if (reason === 'gap') f.step([hand('curl')], 200)
      if (reason === 'departure') f.step([hand('curl', 0.14)])
      if (reason === 'timeout') for (let i = 0; i < 40; i++) f.step([hand('curl')])
      if (reason === 'fist') f.step([hand('fist')])
      if (reason === 'second hand') f.step([hand('curl'), { ...hand('open', 0.3), handedness: 'Right' }])
      if (reason === 'abort') f.gesture.abort()
      f.release()
      expect(f.clicks()).toEqual([])
    }
  )

  it('lets an index bending toward the thumb become a grip without clicking', () => {
    const f = fixture()
    f.curl()
    const pinch = hand('curl')
    pinch.landmarks[4] = { ...pinch.landmarks[8], x: pinch.landmarks[8].x + 0.02 }
    for (let i = 0; i < 3; i++) f.step([pinch])
    expect(f.gesture.phase).toBe('pending')
    f.release()
    expect(f.clicks()).toEqual([])
  })

  it('lets a second hand join a grip after the first index begins bending, without clicking', () => {
    const f = fixture()
    f.curl()
    const left = hand('curl')
    const right = { ...hand('curl', 0.3), handedness: 'Right' }
    for (const h of [left, right]) h.landmarks[4] = { ...h.landmarks[8], x: h.landmarks[8].x + 0.02 }
    for (let i = 0; i < 3; i++) f.step([left, right])
    expect(f.gesture.phase).toBe('scaling')
    expect(f.clicks()).toEqual([])
  })
})
