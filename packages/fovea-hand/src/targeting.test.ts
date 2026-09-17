import { describe, it, expect } from 'vitest'
import { HandTargeting, HoldToPlace, type HandTarget } from './targeting.js'

const a: HandTarget = { id: 'a', label: 'A', rect: { x: 100, y: 100, width: 50, height: 40 }, activation: 'dwell' }
const b: HandTarget = { id: 'b', label: 'B', rect: { x: 190, y: 100, width: 50, height: 40 }, activation: 'dwell' }
function fixture() {
  const targeting = new HandTargeting()
  let now = 0
  const step = (x = 125, y = 120, targets = [a], enabled = true) =>
    targeting.update({ x, y }, (now += 40), targets, enabled)
  return { targeting, step, hold: (n = 25) => Array.from({ length: n }, () => step()) }
}
describe('forgiving hand targeting', () => {
  it('acquires outside the button, tolerates tremor, and activates once until leaving', () => {
    const f = fixture()
    expect(f.step(80).target?.id).toBe('a')
    const updates = Array.from({ length: 60 }, (_, i) => f.step(82 + (i % 3)))
    expect(updates.filter((r) => r.activate)).toHaveLength(1)
    f.step(350)
    expect(f.hold().filter((r) => r.activate)).toHaveLength(1)
  })
  it('does not activate as the hand sweeps through or while pinching', () => {
    const f = fixture()
    for (let i = 0; i < 20; i++) expect(f.step(75 + i * 5).activate).toBeNull()
    for (let i = 0; i < 50; i++) expect(f.step(125, 120, [a], false).progress).toBe(0)
  })
  it('never dwells a control that requires manual adjustment', () => {
    const f = fixture()
    for (let i = 0; i < 60; i++) expect(f.step(125, 120, [{ ...a, activation: 'manual' }]).activate).toBeNull()
  })
  it('offers ambiguous peers without guessing; a nested button outranks its container', () => {
    const f = fixture()
    const result = f.step(170, 120, [a, b])
    expect(result.target).toBeNull()
    expect(result.ambiguous).toHaveLength(2)
    const container = { ...a, id: 'container', rect: { x: 60, y: 50, width: 240, height: 120 } }
    expect(f.step(125, 120, [a, container]).target?.id).toBe('a')
  })
  it('retains a target through small boundary motion instead of flickering between neighbors', () => {
    const f = fixture()
    f.step(145, 120, [a, b])
    expect(f.step(173, 120, [a, b]).target?.id).toBe('a')
    expect(f.step(220, 120, [a, b]).target?.id).toBe('b')
  })
  it('resets on disappearance, movement, stale result gaps, and abort', () => {
    const f = fixture()
    f.hold(15)
    expect(f.step(125, 120, []).target).toBeNull()
    expect(f.step().progress).toBe(0)
    f.hold(15)
    f.targeting.reset()
    expect(f.step().progress).toBe(0)
    f.hold(15)
    expect(f.targeting.update({ x: 125, y: 120 }, 4000, [a]).progress).toBe(0)
    expect(f.targeting.update({ x: 125, y: 120 }, 4040, [{ ...a, rect: { ...a.rect, x: 120 } }]).progress).toBe(0)
  })
  it('blocks a replacement control appearing under an activated target', () => {
    const f = fixture()
    f.hold()
    const replacement = { ...a, id: 'new', label: 'New action' }
    for (let i = 0; i < 40; i++) expect(f.step(125, 120, [replacement]).activate).toBeNull()
  })
  it('honors adjustable dwell time and rejects invalid configuration', () => {
    expect(() => new HandTargeting({ dwellMs: NaN })).toThrow()
    const t = new HandTargeting({ dwellMs: 1000 })
    let activations = 0
    for (let now = 0; now <= 1160; now += 40) activations += Number(!!t.update({ x: 125, y: 120 }, now, [a]).activate)
    expect(activations).toBe(0)
    expect(t.update({ x: 125, y: 120 }, 1200, [a]).activate?.id).toBe('a')
  })
  it('softly attracts at the boundary and preserves precision inside a large header', () => {
    const t = new HandTargeting({ radius: 40, strength: 0.65 })
    const header = { ...a, rect: { x: 100, y: 100, width: 1000, height: 40 } }
    expect(t.update({ x: 60, y: 120 }, 40, [header]).point.x).toBe(60)
    const near = t.update({ x: 80, y: 120 }, 80, [header])
    expect(near.point.x).toBeGreaterThan(80)
    expect(near.point.x).toBeLessThan(100)
    expect(t.update({ x: 125, y: 120 }, 120, [header]).point).toEqual({ x: 125, y: 120 })
  })
  it('turns off nearby attraction and retention at zero strength, while keeping direct dwell', () => {
    const t = new HandTargeting({ radius: 100, strength: 0 })
    expect(t.update({ x: 90, y: 120 }, 40, [a]).target).toBeNull()
    expect(t.update({ x: 101, y: 120 }, 80, [a]).point).toEqual({ x: 101, y: 120 })
    expect(t.update({ x: 99, y: 120 }, 120, [a]).target).toBeNull()
    let activated = false
    for (let now = 160; now <= 1200; now += 40) activated ||= !!t.update({ x: 125, y: 120 }, now, [a]).activate
    expect(activated).toBe(true)
  })
  it('honors range and strength without an abrupt center jump', () => {
    const point = { x: 80, y: 120 }
    expect(new HandTargeting({ radius: 10 }).update(point, 40, [a]).target).toBeNull()
    const weak = new HandTargeting({ radius: 50, strength: 0.2 }).update(point, 40, [a])
    const strong = new HandTargeting({ radius: 50, strength: 1 }).update(point, 40, [a])
    expect(strong.point.x).toBeGreaterThan(weak.point.x)
    expect(strong.point.x).toBeLessThan(125)
    for (const strength of [-1, 2, NaN]) expect(() => new HandTargeting({ strength })).toThrow()
  })
  it('zero range disables both attraction and sticky retention', () => {
    const t = new HandTargeting({ radius: 0, strength: 1 })
    expect(t.update({ x: 101, y: 120 }, 40, [a]).point.x).toBe(101)
    expect(t.update({ x: 99, y: 120 }, 80, [a]).target).toBeNull()
  })
})
describe('relaxed hand placement', () => {
  it('requires movement before a pause can finish and retains the relative offset', () => {
    const h = new HoldToPlace({ x: 200, y: 200 }, 0)
    for (let t = 40; t < 2000; t += 40) expect(h.update({ x: 200, y: 200 }, t).done).toBe(false)
    expect(h.update({ x: 260, y: 240 }, 2000).delta).toEqual({ x: 60, y: 40 })
    let result
    for (let t = 2040; t <= 2920; t += 40) result = h.update({ x: 260, y: 240 }, t)
    expect(result?.done).toBe(true)
  })
  it('does not count missing frames as a successful hold', () => {
    const h = new HoldToPlace({ x: 0, y: 0 }, 0)
    h.update({ x: 100, y: 100 }, 40)
    expect(h.update({ x: 100, y: 100 }, 2000).done).toBe(false)
  })
})
