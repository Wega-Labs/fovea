import type { Bounds, ClientPoint } from "./types.js";

type NormalizedPoint = Readonly<{ x: number; y: number }>;

function validateBounds(name: string, bounds: Bounds): void {
  const values = [bounds.x, bounds.y, bounds.width, bounds.height];
  if (!values.every(Number.isFinite) || bounds.width <= 0 || bounds.height <= 0) {
    throw new RangeError(`${name} must contain finite coordinates and positive dimensions`);
  }
}

export function toClientPoint(
  event: NormalizedPoint,
  displayBounds: Bounds,
  contentBounds: Bounds,
): ClientPoint {
  validateBounds("displayBounds", displayBounds);
  validateBounds("contentBounds", contentBounds);
  if (!Number.isFinite(event.x) || !Number.isFinite(event.y)) {
    throw new RangeError("event coordinates must be finite");
  }
  const screenX = displayBounds.x + event.x * displayBounds.width;
  const screenY = displayBounds.y + event.y * displayBounds.height;
  return { x: screenX - contentBounds.x, y: screenY - contentBounds.y };
}

export function isOffWindow(
  event: NormalizedPoint,
  displayBounds: Bounds,
  contentBounds: Bounds,
): boolean {
  const point = toClientPoint(event, displayBounds, contentBounds);
  return (
    point.x < 0 ||
    point.y < 0 ||
    point.x >= contentBounds.width ||
    point.y >= contentBounds.height
  );
}
