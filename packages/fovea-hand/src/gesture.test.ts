import { describe, expect, it } from 'vitest'
import {
  HandGesture,
  distance,
  mapHandPoint,
  pinchMetric,
  scaleViewport,
  type HandFrame,
  type HandIntent,
  type HandObservation,
  type HandRegion,
  type HandInteractionMode
} from './gesture.js'

function hand(x = 0.5, metric = 0.8, wristX = 0.5, handedness = 'Left'): HandObservation {
  const landmarks = Array.from({ length: 21 }, () => ({ x: wristX, y: 0.6 }))
  landmarks[9] = { x: wristX, y: 0.4 }
  landmarks[8] = { x, y: 0.3 }
  landmarks[4] = { x: x + metric * 0.15, y: 0.3 }
  return { landmarks, handedness }
}

function fixture(region?: HandRegion, mode: HandInteractionMode = 'pinch') {
  const machine = new HandGesture(7, region, mode)
  let time = 1000
  const events: HandIntent[] = []
  const step = (hands: HandObservation[], dt = 40) => {
    time += dt
    const frame: HandFrame = { gen: 7, wallTs: time, ts: time, aspect: 4 / 3, hands }
    const result = machine.step(frame, time, 1001, 801)
    events.push(...result.intents)
    return result.intents
  }
  const pending = () => {
    step([hand()])
    step([hand(0.5, 0.2)])
    step([hand(0.5, 0.2)])
    step([hand(0.5, 0.2)])
    expect(machine.phase).toBe('pending')
  }
  const drag = () => {
    pending()
    step([hand(0.45, 0.2, 0.45)])
    step([hand(0.44, 0.2, 0.44)])
    expect(machine.phase).toBe('dragging')
  }
  return { machine, events, step, pending, drag }
}

describe('hand gestures', () => {
  it('releases a drag with relaxed finger separation, once, without requiring a wide open hand', () => {
    const f = fixture()
    f.drag()
    f.step([hand(0.44, 0.58, 0.44)])
    f.step([hand(0.44, 0.58, 0.44)], 60)
    expect(f.machine.phase).toBe('cooldown')
    expect(f.events.filter((event) => event.type === 'up')).toHaveLength(1)
    for (let i = 0; i < 10; i++) f.step([hand(0.44, 0.58, 0.44)])
    expect(f.events.filter((event) => event.type === 'up')).toHaveLength(1)
  })
  it('defaults to pinch clicks at the frozen aim, with one down/up on release', () => {
    const f = fixture()
    expect(f.machine.interaction).toBe('pinch')
    f.pending()
    f.step([hand(0.46, 0.8)])
    expect(f.step([hand(0.44, 0.8)], 60)).toEqual([
      { type: 'down', x: 500, y: expect.closeTo(200, 8) },
      { type: 'up', x: 500, y: expect.closeTo(200, 8) }
    ])
  })
  it('requires an open hand before arming', () => {
    const f = fixture()
    for (let i = 0; i < 8; i++) f.step([hand(0.5, 0.2)])
    expect(f.machine.phase).toBe('idle')
    expect(f.events).toEqual([])
    f.step([hand()])
    expect(f.machine.phase).toBe('armed')
  })
  it('confirms a grip with elapsed time and releases a stationary pinch without clicking (curl mode)', () => {
    const f = fixture(undefined, 'curl')
    f.step([hand()])
    f.step([hand(0.5, 0.2)], 1)
    for (let i = 0; i < 20; i++) f.step([hand(0.5, 0.2)], 1)
    expect(f.machine.phase).toBe('armed')
    f.step([hand(0.5, 0.4)])
    f.step([hand(0.5, 0.2)])
    f.step([hand(0.5, 0.2)], 80)
    expect(f.machine.phase).toBe('pending')
    expect(f.events.filter((e) => e.type === 'down')).toHaveLength(0)
    f.step([hand()])
    expect(f.step([hand()], 59)).toEqual([])
    expect(f.step([hand()], 1)).toEqual([])
  })
  it('opening-only motion cannot turn a stationary grip into a click (curl mode)', () => {
    const f = fixture(undefined, 'curl')
    f.pending()
    f.step([hand(0.46, 0.36)])
    f.step([hand(0.43, 0.48)])
    f.step([hand(0.4, 0.8)])
    f.step([hand(0.39, 0.8)], 60)
    expect(f.events.filter((e) => e.type === 'down' || e.type === 'up' || e.type === 'click')).toEqual([])
  })
  it('resets smoothing at pinch confirmation', () => {
    const f = fixture()
    f.step([hand(0.7)])
    f.step([hand(0.6, 0.2)])
    f.step([hand(0.5, 0.2)])
    f.step([hand(0.4, 0.2)])
    for (let i = 0; i < 10; i++) f.step([hand(0.4, 0.2)])
    expect(f.machine.phase).toBe('pending')
  })
  it('finger movement during a stationary pinch never clicks (curl mode)', () => {
    const f = fixture(undefined, 'curl')
    f.step([hand(0.5)])
    f.step([hand(0.46, 0.2)])
    f.step([hand(0.44, 0.2)])
    f.step([hand(0.42, 0.2)])
    f.step([hand(0.4, 0.8)])
    f.step([hand(0.39, 0.8)], 60)
    expect(f.events.filter((e) => e.type === 'down' || e.type === 'up' || e.type === 'click')).toEqual([])
  })
  it('starts at the first deliberate palm movement after confirmation and preserves the press offset', () => {
    const f = fixture()
    f.pending()
    const intents = f.step([hand(0.45, 0.2, 0.45)])
    expect(f.machine.phase).toBe('dragging')
    expect(intents).toEqual([
      { type: 'down', x: 500, y: expect.closeTo(200, 8) },
      { type: 'move', x: expect.any(Number), y: expect.closeTo(200, 8) }
    ])
    const moved = intents[1] as { type: 'move'; x: number; y: number }
    expect(moved.x).toBeGreaterThan(530)
    expect(moved.x).toBeLessThan(575)
    const opening = f.step([hand(0.44, 0.8, 0.44)])[0]
    expect(opening.type).toBe('move')
    expect(f.step([hand(0.44, 0.8, 0.44)], 60)).toEqual([{ ...opening, type: 'up' }])
  })
  it('rejects a small single-sample palm spike during a stationary pinch', () => {
    const f = fixture()
    f.pending()
    f.step([hand(0.4964, 0.2, 0.4964)], 1000 / 30)
    f.step([hand(0.5, 0.2, 0.5)], 1000 / 30)
    expect(f.machine.phase).toBe('pending')
    expect(f.events.some((event) => event.type === 'down')).toBe(false)
  })
  it.each([60, 180, 600])('overlaps pinch confirmation with %ipx/s movement at 30Hz', (speed) => {
    const f = fixture()
    const dt = 1000 / 30
    f.step([hand()], dt)
    f.step([hand(0.5, 0.2)], dt)
    let startedAt = Infinity
    for (let frame = 1; frame <= 6; frame++) {
      const elapsed = frame * dt
      const x = 0.5 - (speed * elapsed) / 1000 / 1250
      const intents = f.step([hand(x, 0.2, x)], dt)
      if (intents.some((intent) => intent.type === 'down')) {
        startedAt = elapsed
        expect(intents[0]).toEqual({ type: 'down', x: 500, y: expect.closeTo(200, 8) })
        expect((intents[1] as { x: number }).x).toBeGreaterThan(502)
        break
      }
    }
    expect(startedAt).toBeGreaterThanOrEqual(40)
    expect(startedAt).toBeLessThanOrEqual(speed === 60 ? 100.01 : 67)
  })
  it('keeps dragging through a transient release candidate and releases exactly once after a deliberate open', () => {
    const f = fixture()
    f.drag()
    for (const [metric, x] of [
      [0.68, 0.43],
      [0.51, 0.42],
      [0.49, 0.41]
    ]) {
      expect(f.step([hand(x, metric, x)], 1000 / 30)[0].type).toBe('move')
      expect(f.machine.phase).toBe('dragging')
    }
    expect(f.step([hand(0.4, 0.8, 0.4)])[0].type).toBe('move')
    const lastMove = f.step([hand(0.39, 0.8, 0.39)], 59)[0]
    expect(lastMove.type).toBe('move')
    expect(f.step([hand(0.39, 0.8, 0.39)], 1)).toEqual([{ ...lastMove, type: 'up' }])
    expect(f.events.filter((event) => event.type === 'down')).toHaveLength(1)
    expect(f.events.filter((event) => event.type === 'up')).toHaveLength(1)
  })
  it.each([151, 250, 399])('rebases a held drag after a fresh %ims inference gap without replay or release', (gap) => {
    const f = fixture()
    f.drag()
    const lastMove = f.events.at(-1) as { type: 'move'; x: number; y: number }
    expect(f.step([hand(0.35, 0.2, 0.35)], gap)).toEqual([lastMove])
    expect(f.machine.phase).toBe('dragging')
    const next = f.step([hand(0.34, 0.2, 0.34)])[0] as { type: 'move'; x: number; y: number }
    expect(next.x).toBeGreaterThan(lastMove.x)
    expect(next.x).toBeLessThan(lastMove.x + 12.5)
    expect(f.events.filter((event) => event.type === 'down')).toHaveLength(1)
  })
  it.each([400, 401])('cancels a drag once after %ims tracking loss and requires an open hand to rearm', (gap) => {
    const f = fixture()
    f.drag()
    expect(f.step([hand(0.35, 0.2, 0.35)], gap)).toEqual([{ type: 'cancel' }])
    expect(f.machine.phase).toBe('idle')
    expect(f.step([hand(0.34, 0.2, 0.34)])).toEqual([])
    expect(f.machine.phase).toBe('idle')
  })
  it('a second confirmed pinch joins pending or dragging, releasing a native drag before scaling', () => {
    for (const phase of ['pending', 'dragging']) {
      const f = fixture()
      if (phase === 'pending') f.pending()
      else f.drag()
      for (let i = 0; i < 3; i++) f.step([hand(0.5, 0.2), hand(0.8, 0.2, 0.8, 'Right')])
      expect(f.machine.phase).toBe('scaling')
      const start = f.events.findIndex((e) => e.type === 'scale' && e.phase === 'start')
      if (phase === 'pending') expect(f.events.filter((e) => e.type === 'down')).toHaveLength(0)
      else expect(f.events[start - 1]).toEqual({ type: 'cancel' })
      expect(f.machine.abort()).toEqual([expect.objectContaining({ type: 'scale', phase: 'cancel' })])
    }
  })
  it('lets slightly staggered two-hand pinches confirm before beginning a native drag', () => {
    const f = fixture({ left: 0.32, top: 0.32, right: 0.68, bottom: 0.68 })
    f.step([hand(0.4, 0.8, 0.4), hand(0.7, 0.8, 0.7)])
    f.step([hand(0.4, 0.2, 0.4), hand(0.7, 0.8, 0.7)])
    f.step([hand(0.39, 0.2, 0.39), hand(0.71, 0.2, 0.71)])
    expect(f.machine.phase).toBe('pending')
    f.step([hand(0.38, 0.2, 0.38), hand(0.72, 0.2, 0.72)])
    expect(f.machine.phase).toBe('scaling')
    expect(f.events.some((e) => e.type === 'down')).toBe(false)
  })
  it('keeps two-hand scale geometry independent of pointing reach and unclipped at its edges', () => {
    const scales = [undefined, { left: 0.32, top: 0.32, right: 0.68, bottom: 0.68 }].map((region) => {
      const f = fixture(region)
      f.step([hand(0.1, 0.8, 0.1), hand(0.3, 0.8, 0.3)])
      for (let i = 0; i < 3; i++) f.step([hand(0.1, 0.2, 0.1), hand(0.3, 0.2, 0.3)])
      const start = f.events.find((e) => e.type === 'scale' && e.phase === 'start')!
      for (let i = 0; i < 12; i++) f.step([hand(0.08, 0.2, 0.08), hand(0.32, 0.2, 0.32)])
      const end = f.events.at(-1)!
      if (start.type !== 'scale' || end.type !== 'scale') throw Error('Expected scale')
      return { start: distance(...start.points), end: distance(...end.points) }
    })
    expect(scales[1].start).toBeCloseTo(scales[0].start, 6)
    expect(scales[1].end).toBeCloseTo(scales[0].end, 1)
    expect(scales[1].start).toBeGreaterThan(200)
  })
  it('ends scaling when either hand opens and cooldown waits for both', () => {
    const f = fixture()
    f.pending()
    for (let i = 0; i < 3; i++) f.step([hand(0.5, 0.2), hand(0.8, 0.2, 0.8)])
    f.step([hand(), hand(0.8, 0.2, 0.8)])
    f.step([hand(), hand(0.8, 0.2, 0.8)], 60)
    expect(f.machine.phase).toBe('cooldown')
    f.step([hand(), hand(0.8, 0.8, 0.8)])
    expect(f.machine.phase).toBe('cooldown')
    f.step([hand(), hand(0.8, 0.8, 0.8)], 60)
    expect(f.machine.phase).toBe('armed')
  })
  it('loss cancels a drag exactly once, drops pending silently, and requires a new open hand', () => {
    const f = fixture()
    f.pending()
    expect(f.step([])).toEqual([])
    f.drag()
    expect(f.step([])).toEqual([{ type: 'cancel' }])
    expect(f.machine.abort()).toEqual([])
    for (let i = 0; i < 5; i++) f.step([hand(0.5, 0.2)])
    expect(f.machine.phase).toBe('idle')
  })
  it('keeps identity through result reordering and handedness flips', () => {
    const f = fixture()
    f.drag()
    const events = f.step([hand(0.8, 0.8, 0.8, 'Left'), hand(0.44, 0.2, 0.44, 'Right')])
    expect(f.machine.phase).toBe('dragging')
    expect(events[0]).toEqual(expect.objectContaining({ type: 'move', x: expect.any(Number) }))
    expect((events[0] as { x: number }).x).toBeGreaterThan(550)
    expect(f.step([hand(0.8, 0.2, 0.9)])).toEqual([{ type: 'cancel' }])
  })
  it('rejects stale generations, stale ages, future and reordered results', () => {
    const machine = new HandGesture(7)
    const frame = { gen: 7, wallTs: 1000, ts: 1, aspect: 4 / 3, hands: [hand()] }
    expect(machine.step({ ...frame, gen: 6 }, 1000, 1000, 800).accepted).toBe(false)
    expect(machine.step(frame, 1151, 1000, 800).accepted).toBe(false)
    expect(machine.step(frame, 999, 1000, 800).accepted).toBe(false)
    expect(machine.step(frame, 1100, 1000, 800).accepted).toBe(true)
    expect(machine.step(frame, 1100, 1000, 800).accepted).toBe(false)
  })
  it('does not interpret fingertip curl after pinch confirmation as palm dragging (curl mode)', () => {
    const f = fixture(undefined, 'curl')
    f.pending()
    for (let i = 0; i < 8; i++) f.step([hand(0.4 - i * 0.01, 0.2)])
    expect(f.machine.phase).toBe('pending')
    expect(f.events.filter((event) => event.type === 'down')).toEqual([])
    f.step([hand(0.3, 0.8)])
    expect(f.step([hand(0.3, 0.8)], 60)).toEqual([])
  })
  it('drags smoothly when the palm lies outside a compact fingertip reach region', () => {
    const f = fixture({ left: 0.3, right: 0.7, top: 0.15, bottom: 0.4 })
    f.pending()
    expect(f.machine.sample!.palm.y).toBeGreaterThan(800)
    const translate = (dy: number): HandObservation => {
      const observation = hand(0.5, 0.2)
      observation.landmarks = observation.landmarks.map((point) => ({ x: point.x, y: point.y + dy }))
      return observation
    }
    const intents = f.step([translate(-0.02)])
    expect(f.machine.phase).toBe('dragging')
    expect(intents).toEqual([
      { type: 'down', x: expect.closeTo(500, 8), y: expect.closeTo(480, 8) },
      { type: 'move', x: expect.closeTo(500, 8), y: expect.any(Number) }
    ])
    expect((intents[1] as { y: number }).y).toBeLessThan(450)
    expect(f.machine.sample!.palm.y).toBeGreaterThan(800)
    expect(f.machine.sample!.point.y).toBeGreaterThanOrEqual(0)
    expect(f.machine.sample!.point.y).toBeLessThanOrEqual(800)
  })
  it('keeps a committed drag moving through a relaxed pinch without freezing', () => {
    const f = fixture()
    f.drag()
    let previousX = 550
    for (let i = 0; i < 8; i++) {
      const wristX = 0.42 - i * 0.015
      const events = f.step([hand(wristX + 0.03, i % 2 ? 0.49 : 0.51, wristX)])
      expect(f.machine.phase).toBe('dragging')
      expect(events).toHaveLength(1)
      expect(events[0].type).toBe('move')
      const x = (events[0] as { x: number }).x
      expect(x).toBeGreaterThan(previousX)
      previousX = x
    }
    const lastMove = f.step([hand(0.15, 0.8, 0.315)])[0]
    expect(lastMove.type).toBe('move')
    expect(f.step([hand(0.1, 0.8, 0.315)], 60)).toEqual([{ ...lastMove, type: 'up' }])
    expect(f.events.filter((event) => event.type === 'down')).toHaveLength(1)
    expect(f.events.filter((event) => event.type === 'up')).toHaveLength(1)
  })
  it('uses stable palms for scaling and advances through the relaxed pinch band', () => {
    const f = fixture()
    f.pending()
    for (let i = 0; i < 3; i++) f.step([hand(0.5, 0.2), hand(0.8, 0.2, 0.8)])
    const before = f.events.find((event) => event.type === 'scale' && event.phase === 'start')
    expect(before?.type).toBe('scale')
    const curl = f.step([hand(0.4, 0.43), hand(0.9, 0.43, 0.8)])[0]
    expect(curl.type).toBe('scale')
    if (before?.type !== 'scale' || curl.type !== 'scale') throw new Error('Expected scale')
    expect(distance(...curl.points)).toBeCloseTo(distance(...before.points), 8)
    const moved = f.step([hand(0.38, 0.51, 0.48), hand(0.92, 0.49, 0.82)])[0]
    expect(moved.type).toBe('scale')
    if (before?.type !== 'scale' || moved.type !== 'scale') throw new Error('Expected scale')
    expect(distance(...moved.points)).toBeGreaterThan(distance(...before.points))
    let span = distance(...moved.points)
    for (let i = 1; i <= 4; i++) {
      const next = f.step([hand(0.38, i % 2 ? 0.49 : 0.51, 0.48 - i * 0.01), hand(0.92, 0.51, 0.82 + i * 0.01)])[0]
      expect(next.type).toBe('scale')
      if (next.type !== 'scale') throw new Error('Expected scale')
      expect(distance(...next.points)).toBeGreaterThan(span)
      span = distance(...next.points)
    }
    expect(f.step([hand(0.38, 0.8, 0.48), hand(0.92, 0.43, 0.82)])[0]).toEqual(
      expect.objectContaining({ type: 'scale', phase: 'move' })
    )
    expect(f.step([hand(0.38, 0.8, 0.48), hand(0.92, 0.43, 0.82)], 60)).toEqual([
      expect.objectContaining({ type: 'scale', phase: 'end' })
    ])
  })
  it('balances every native press over repeated abort/release cycles', () => {
    const f = fixture()
    for (let i = 0; i < 10; i++) {
      f.drag()
      if (i % 2) {
        f.step([hand()])
        f.step([hand()], 80)
        f.step([])
      } else f.step([])
    }
    let held = false
    for (const event of f.events) {
      if (event.type === 'down') {
        expect(held).toBe(false)
        held = true
      }
      if (event.type === 'up' || event.type === 'cancel') {
        expect(held).toBe(true)
        held = false
      }
    }
    expect(held).toBe(false)
  })
})

describe('mapping and scale math', () => {
  it('mirrors and clamps the active region', () => {
    expect(mapHandPoint({ x: 0.1, y: 0.1 }, 1001, 801)).toEqual({ x: 1000, y: 0 })
    expect(mapHandPoint({ x: 1, y: 1 }, 1001, 801)).toEqual({ x: 0, y: 800 })
    expect(pinchMetric(hand(0.5, 0.2).landmarks, 4 / 3)).toBeCloseTo(0.2)
  })
  it('preserves the pane-relative anchor (including sidebar inset) and clamps zoom', () => {
    const start = { x: -30, y: 50, zoom: 0.8 }
    const anchor = { x: 700 - 240, y: 300 - 48 }
    const before = { x: (anchor.x - start.x) / start.zoom, y: (anchor.y - start.y) / start.zoom }
    for (const d of [0, 50, 200, 10000]) {
      const next = scaleViewport(start, anchor, 100, d)
      expect(next.zoom).toBeGreaterThanOrEqual(0.05)
      expect(next.zoom).toBeLessThanOrEqual(2)
      expect(distance(before, { x: (anchor.x - next.x) / next.zoom, y: (anchor.y - next.y) / next.zoom })).toBeLessThan(
        1e-8
      )
    }
  })
})
