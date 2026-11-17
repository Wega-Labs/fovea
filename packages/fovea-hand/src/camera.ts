import type { HandFrame } from './gesture.js'
import type { TrackerRequest, TrackerResult } from './tracker.worker.js'

export interface HandCameraCallbacks {
  onFrame(frame: HandFrame): void
  onReady(): void
  onError(message: string): void
}

declare global {
  interface Window {
    MediaStreamTrackProcessor: new (options: { track: MediaStreamTrack; maxBufferSize: number }) => {
      readable: ReadableStream<VideoFrame>
    }
  }
}

export function handCameraError(error: unknown): string {
  if (error instanceof DOMException) {
    if (error.name === 'NotAllowedError' || error.name === 'SecurityError') return 'Camera permission denied'
    if (error.name === 'NotFoundError') return 'No camera found'
    if (error.name === 'NotReadableError') return 'Camera is unavailable or already in use'
  }
  return error instanceof Error ? error.message : 'Hand tracking could not start'
}

/** One camera acquisition. stop() is safe before permission resolves and is idempotent. */
export class HandCamera {
  private stopped = false
  private started = false
  private stream?: MediaStream
  private worker?: Worker

  constructor(
    private readonly gen: number,
    private readonly callbacks: HandCameraCallbacks
  ) {}

  async start(): Promise<void> {
    if (this.started || this.stopped) return
    this.started = true
    try {
      if (!navigator.mediaDevices?.getUserMedia || !window.MediaStreamTrackProcessor)
        throw new Error('This browser does not support camera frame processing')
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, frameRate: 30, facingMode: 'user' }
      })
      if (this.stopped) {
        stream.getTracks().forEach((track) => track.stop())
        return
      }
      this.stream = stream
      const track = stream.getVideoTracks()[0]
      if (!track) throw new Error('No camera video track')
      track.addEventListener('ended', () => this.fail('Camera disconnected'))
      track.addEventListener('mute', () => this.fail('Camera stopped providing video'))
      const processor = new window.MediaStreamTrackProcessor({ track, maxBufferSize: 1 })
      const worker = new Worker(new URL('./tracker.worker.js', import.meta.url), { type: 'module' })
      this.worker = worker
      worker.onmessage = (event: MessageEvent<TrackerResult>) => {
        if (this.stopped || event.data.gen !== this.gen) return
        if (event.data.type === 'ready') this.callbacks.onReady()
        else if (event.data.type === 'frame') this.callbacks.onFrame(event.data)
        else this.fail(`Hand tracking unavailable: ${event.data.message}`)
      }
      worker.onerror = (event) => {
        event.preventDefault()
        this.fail('Hand tracking worker failed')
      }
      worker.onmessageerror = () => this.fail('Hand tracking worker communication failed')
      worker.postMessage({ type: 'start', gen: this.gen, readable: processor.readable } satisfies TrackerRequest, [
        processor.readable
      ])
    } catch (error) {
      this.fail(handCameraError(error))
    }
  }

  stop(): void {
    if (this.stopped) return
    this.stopped = true
    const worker = this.worker
    this.worker = undefined
    if (worker) {
      worker.onmessage = worker.onerror = worker.onmessageerror = null
      try {
        worker.postMessage({ type: 'stop' } satisfies TrackerRequest)
      } catch {
        /* The worker may already have exited. */
      }
      setTimeout(() => worker.terminate(), 200)
    }
    this.stream?.getTracks().forEach((track) => track.stop())
    this.stream = undefined
  }

  private fail(message: string): void {
    if (this.stopped) return
    this.stop()
    this.callbacks.onError(message)
  }
}
