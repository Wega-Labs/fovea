import type { HandObservation } from './gesture.js'

/** Bent middle finger against the thumb, relaxed index nearby, ring/pinky folded.
 * The release folds the middle into the palm instead of straightening it. */
export function snapHand(
  pose: 'contact' | 'release' | 'open' | 'pinch' | 'separating' = 'contact',
  dx = 0
): HandObservation {
  const landmarks = [
    [0.5, 0.78],
    [0.4, 0.7],
    [0.37, 0.63],
    [0.36, 0.54],
    [0.42, 0.48],
    [0.42, 0.58],
    [0.37, 0.43],
    [0.39, 0.39],
    [0.43, 0.405],
    [0.5, 0.58],
    [0.5, 0.43],
    [0.465, 0.435],
    [0.43, 0.485],
    [0.58, 0.6],
    [0.63, 0.49],
    [0.63, 0.58],
    [0.58, 0.67],
    [0.64, 0.64],
    [0.7, 0.55],
    [0.7, 0.62],
    [0.65, 0.7]
  ].map(([x, y]) => ({ x: x + dx, y }))
  if (pose === 'release') {
    landmarks[4] = { x: 0.36 + dx, y: 0.45 }
    landmarks[10] = { x: 0.5 + dx, y: 0.48 }
    landmarks[11] = { x: 0.52 + dx, y: 0.59 }
    landmarks[12] = { x: 0.51 + dx, y: 0.68 }
  }
  if (pose === 'separating') landmarks[12] = { x: 0.49 + dx, y: 0.48 }
  if (pose === 'open' || pose === 'pinch') {
    landmarks[4] = { x: 0.32 + dx, y: 0.52 }
    for (const base of [5, 9, 13, 17])
      for (let joint = 1; joint <= 3; joint++)
        landmarks[base + joint] = { x: landmarks[base].x, y: landmarks[base].y - joint * 0.085 }
    if (pose === 'pinch') landmarks[4] = { x: landmarks[8].x + 0.015, y: landmarks[8].y }
  }
  return { handedness: dx ? 'Right' : 'Left', landmarks }
}
