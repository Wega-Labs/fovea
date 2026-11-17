import { OneEuroFilter } from '1eurofilter'
import type { HandPoint } from './gesture.js'

/** Parameters are in CSS pixels and seconds, independent of camera resolution and frame rate. */
export class SmoothPoint {
  private readonly x = new OneEuroFilter(30, 1.5, 0.04, 1)
  private readonly y = new OneEuroFilter(30, 1.5, 0.04, 1)
  filter(point: HandPoint, milliseconds: number): HandPoint {
    return { x: this.x.filter(point.x, milliseconds / 1000), y: this.y.filter(point.y, milliseconds / 1000) }
  }
  reset(): void {
    this.x.reset()
    this.y.reset()
  }
}
