// @vitest-environment jsdom
import { beforeEach, afterEach, describe, it, expect, vi } from 'vitest'
import { DOMHandTargets } from './dom-targets.js'

describe('host DOM targets', () => {
  let button: HTMLButtonElement, hit: Element | null
  beforeEach(() => {
    button = document.createElement('button')
    button.textContent = 'Open'
    document.body.append(button)
    hit = button
    vi.spyOn(button, 'getBoundingClientRect').mockReturnValue({
      x: 10,
      y: 20,
      left: 10,
      top: 20,
      right: 110,
      bottom: 60,
      width: 100,
      height: 40,
      toJSON: () => ({})
    })
    Object.defineProperty(document, 'elementFromPoint', { configurable: true, value: () => hit })
  })
  afterEach(() => {
    document.body.innerHTML = ''
    vi.restoreAllMocks()
  })
  const policy = () => ({ label: 'Open', activation: 'dwell' as const })
  it('does not register occluded, disabled, or inert controls', () => {
    const dom = new DOMHandTargets()
    expect(dom.collect([button], policy)).toHaveLength(1)
    hit = document.body
    expect(dom.collect([button], policy)).toHaveLength(0)
    hit = button
    button.disabled = true
    expect(dom.collect([button], policy)).toHaveLength(0)
    button.disabled = false
    button.setAttribute('inert', '')
    expect(dom.collect([button], policy)).toHaveLength(0)
  })
  it('revalidates a target at activation time and retires changed meanings', () => {
    const dom = new DOMHandTargets(),
      first = dom.collect([button], policy)[0]
    expect(dom.resolve(first.id)).toEqual({ x: 60, y: 40 })
    button.disabled = true
    expect(dom.resolve(first.id)).toBeNull()
    button.disabled = false
    const second = dom.collect([button], () => ({ label: 'Delete', activation: 'manual' }))[0]
    expect(first.id).not.toBe(second.id)
    expect(dom.resolve(first.id)).toBeNull()
    button.remove()
    expect(dom.resolve(second.id)).toBeNull()
  })
  it('does not infer activation permission from button text', () => {
    const dom = new DOMHandTargets()
    expect(dom.collect([button], () => null)).toEqual([])
  })
  it('refreshes one target with one geometry read and keeps unrelated captures', () => {
    const dom = new DOMHandTargets()
    const first = dom.capture(button, policy)!
    const other = document.createElement('button')
    document.body.append(other)
    vi.spyOn(other, 'getBoundingClientRect').mockReturnValue(button.getBoundingClientRect())
    hit = other
    const second = dom.capture(other, policy)!
    hit = button
    const geometry = vi.mocked(button.getBoundingClientRect)
    geometry.mockClear()
    const refreshed = dom.capture(button, policy)
    expect(refreshed).toEqual(first)
    expect(geometry).toHaveBeenCalledOnce()
    expect(dom.element(second.id)).toBe(other)
  })
  it('retires a captured id when its action key changes even with an identical label', () => {
    const dom = new DOMHandTargets()
    const first = dom.capture(button, () => ({ ...policy(), key: 'open' }))!
    const changed = dom.capture(button, () => ({ ...policy(), key: 'delete' }))!
    expect(changed.id).not.toBe(first.id)
    expect(dom.element(first.id)).toBeNull()
    expect(dom.element(changed.id)).toBe(button)
  })
  it('retires a locally revalidated target when permission or visibility is lost', () => {
    const dom = new DOMHandTargets()
    let captured = dom.capture(button, policy)!
    expect(dom.capture(button, () => null)).toBeNull()
    expect(dom.element(captured.id)).toBeNull()
    captured = dom.capture(button, policy)!
    hit = document.body
    expect(dom.capture(button, policy)).toBeNull()
    expect(dom.element(captured.id)).toBeNull()
    hit = button
    captured = dom.capture(button, policy)!
    button.remove()
    expect(dom.capture(button, policy)).toBeNull()
    expect(dom.element(captured.id)).toBeNull()
  })
})
