// @vitest-environment jsdom
import { afterEach, beforeEach, it, expect, vi } from 'vitest'
import { HandCamera } from './camera.js'
let resolveCamera: (stream: unknown) => void
const stopped = vi.fn(),
  error = vi.fn(),
  ready = vi.fn(),
  frame = vi.fn()
class WorkerMock {
  static latest: WorkerMock
  onmessage: ((event: MessageEvent) => void) | null = null
  onerror: ((event: ErrorEvent) => void) | null = null
  onmessageerror: (() => void) | null = null
  terminate = vi.fn()
  postMessage = vi.fn()
  constructor() {
    WorkerMock.latest = this
  }
}
beforeEach(() => {
  vi.useFakeTimers()
  vi.clearAllMocks()
  vi.stubGlobal('Worker', WorkerMock)
  vi.stubGlobal(
    'MediaStreamTrackProcessor',
    class {
      readable = new ReadableStream()
    }
  )
  Object.defineProperty(navigator, 'mediaDevices', {
    configurable: true,
    value: {
      getUserMedia: () =>
        new Promise((resolve) => {
          resolveCamera = resolve
        })
    }
  })
})
afterEach(() => {
  vi.runOnlyPendingTimers()
  vi.useRealTimers()
  vi.unstubAllGlobals()
})
const stream = () => {
  const track = Object.assign(new EventTarget(), { stop: stopped })
  return { getVideoTracks: () => [track], getTracks: () => [track] }
}
it('stops a late camera acquisition and permits no frames after stop', async () => {
  const camera = new HandCamera(7, { onError: error, onReady: ready, onFrame: frame })
  const start = camera.start()
  camera.stop()
  resolveCamera(stream())
  await start
  expect(stopped).toHaveBeenCalledOnce()
  expect(ready).not.toHaveBeenCalled()
})
it('filters generations and tears down on worker error', async () => {
  const camera = new HandCamera(7, { onError: error, onReady: ready, onFrame: frame })
  const start = camera.start()
  resolveCamera(stream())
  await start
  const worker = WorkerMock.latest
  worker.onmessage?.({ data: { type: 'ready', gen: 6 } } as MessageEvent)
  expect(ready).not.toHaveBeenCalled()
  worker.onmessage?.({ data: { type: 'ready', gen: 7 } } as MessageEvent)
  expect(ready).toHaveBeenCalledOnce()
  worker.onmessage?.({ data: { type: 'error', gen: 7, message: 'GPU failed' } } as MessageEvent)
  expect(error).toHaveBeenCalledWith('Hand tracking unavailable: GPU failed')
  expect(stopped).toHaveBeenCalledOnce()
  camera.stop()
  vi.advanceTimersByTime(200)
  expect(worker.terminate).toHaveBeenCalledOnce()
})
