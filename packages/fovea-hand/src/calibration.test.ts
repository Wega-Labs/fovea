import { it, expect } from 'vitest'
import { HandReachCalibration } from './calibration.js'
import { mapHandPoint } from './gesture.js'
it('fits a comfortable reach while rejecting stationary or insufficient samples', () => {
  const still = new HandReachCalibration(0)
  for (let i = 0; i < 100; i++) still.add({ x: 0.5, y: 0.5 })
  expect(still.finish()).toBeNull()
  const sweep = new HandReachCalibration(0)
  for (let i = 0; i < 100; i++) sweep.add({ x: 0.3 + (i % 10) * 0.04, y: 0.2 + Math.floor(i / 10) * 0.05 })
  const region = sweep.finish()!
  expect(region.left).toBeGreaterThan(0.2)
  expect(region.right).toBeLessThan(0.8)
  expect(mapHandPoint({ x: region.left, y: region.top }, 1001, 801, region)).toEqual({ x: 1000, y: 0 })
})
