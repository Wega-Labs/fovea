import type { HandPoint } from './gesture.js'
import type { HandTarget } from './targeting.js'

export interface DOMTargetPolicy {
  label: string
  activation: HandTarget['activation']
  /** Host identity for actions whose label can stay the same while their meaning changes. */
  key?: string
}

/** Opt-in targets only. The host owns which elements are safe to activate by pausing. */
export class DOMHandTargets {
  private ids = new WeakMap<Element, string>()
  private elements = new Map<string, HTMLElement>()
  private meanings = new WeakMap<Element, string>()
  private sequence = 0

  collect(elements: Iterable<HTMLElement>, policy: (element: HTMLElement) => DOMTargetPolicy | null): HandTarget[] {
    this.elements.clear()
    const targets: HandTarget[] = []
    for (const element of elements) {
      const target = this.capture(element, policy)
      if (target) targets.push(target)
    }
    return targets
  }

  /** Refresh one captured control without rediscovering unrelated controls. */
  capture(element: HTMLElement, policy: (element: HTMLElement) => DOMTargetPolicy | null): HandTarget | null {
    const info = policy(element)
    const geometry = info ? this.visibleGeometry(element) : null
    let id = this.ids.get(element)
    if (!info || !geometry) {
      if (id) this.elements.delete(id)
      this.ids.delete(element)
      this.meanings.delete(element)
      return null
    }
    const meaning = JSON.stringify([info.activation, info.label, info.key ?? null])
    if (!id || this.meanings.get(element) !== meaning) {
      if (id) this.elements.delete(id)
      id = `dom-${++this.sequence}`
      this.ids.set(element, id)
      this.meanings.set(element, meaning)
    }
    this.elements.set(id, element)
    return { id, label: info.label, activation: info.activation, ...geometry }
  }

  element(id: string): HTMLElement | null {
    return this.elements.get(id) ?? null
  }

  /** Recheck at activation time: a dialog, disabled state, or navigation may have intervened. */
  resolve(id: string): HandPoint | null {
    const element = this.elements.get(id)
    return element ? this.visiblePoint(element) : null
  }

  visiblePoint(element: HTMLElement): HandPoint | null {
    return this.visibleGeometry(element)?.point ?? null
  }

  private visibleGeometry(element: HTMLElement): Pick<HandTarget, 'point' | 'rect'> | null {
    if (
      !element.isConnected ||
      element.closest('[inert], [hidden], [aria-hidden="true"], [aria-disabled="true"]') ||
      element.matches(':disabled')
    )
      return null
    const style = getComputedStyle(element)
    if (style.display === 'none' || style.visibility !== 'visible' || style.opacity === '0') return null
    const rect = element.getBoundingClientRect()
    const left = Math.max(0, rect.left),
      top = Math.max(0, rect.top),
      right = Math.min(innerWidth, rect.right),
      bottom = Math.min(innerHeight, rect.bottom)
    if (right - left < 2 || bottom - top < 2) return null
    const point = { x: (left + right) / 2, y: (top + bottom) / 2 }
    const hit = document.elementFromPoint(point.x, point.y)
    const nested = hit?.closest('button, a[href], input, textarea, select, [role="button"]')
    if (nested && nested !== element && element.contains(nested)) return null
    return hit && (hit === element || element.contains(hit))
      ? { point, rect: { x: left, y: top, width: right - left, height: bottom - top } }
      : null
  }
}
