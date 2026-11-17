import { HandLandmarker } from '@mediapipe/tasks-vision'
import loaderUrl from '@mediapipe/tasks-vision/vision_wasm_module_internal.js?url'
import wasmUrl from '@mediapipe/tasks-vision/vision_wasm_module_internal.wasm?url'
import modelUrl from './hand_landmarker.task?url'
import type { HandFrame } from './gesture.js'

export type TrackerRequest = { type: 'start'; gen: number; readable: ReadableStream<VideoFrame> } | { type: 'stop' }
export type TrackerResult =
  | ({ type: 'frame' } & HandFrame)
  | { type: 'ready'; gen: number }
  | { type: 'error'; gen: number; message: string }

let stopped = false
let reader: ReadableStreamDefaultReader<VideoFrame> | undefined

const post = (message: TrackerResult): void => self.postMessage(message)

async function track(gen: number, readable: ReadableStream<VideoFrame>): Promise<void> {
  let landmarker: HandLandmarker | undefined
  reader = readable.getReader()
  try {
    landmarker = await HandLandmarker.createFromOptions(
      { wasmLoaderPath: loaderUrl, wasmBinaryPath: wasmUrl },
      {
        baseOptions: { modelAssetPath: modelUrl, delegate: 'GPU' },
        runningMode: 'VIDEO',
        numHands: 2,
        canvas: new OffscreenCanvas(640, 480)
      }
    )
    if (stopped) return
    post({ type: 'ready', gen })
    let previousTs = -1
    while (!stopped) {
      const { value: frame, done } = await reader.read()
      if (done) break
      try {
        if (stopped) break
        const receipt = performance.now()
        const wallTs = performance.timeOrigin + receipt
        const ts = Math.max(receipt, previousTs + 0.001)
        previousTs = ts
        const result = landmarker.detectForVideo(frame, ts)
        post({
          type: 'frame',
          gen,
          wallTs,
          ts,
          aspect: frame.displayWidth / frame.displayHeight,
          hands: result.landmarks.map((landmarks, index) => ({
            landmarks: landmarks.map(({ x, y }) => ({ x, y })),
            handedness: result.handedness[index]?.[0]?.categoryName ?? ''
          }))
        })
      } finally {
        frame.close()
      }
    }
    if (!stopped) post({ type: 'error', gen, message: 'Camera stream ended' })
  } catch (error) {
    if (!stopped) post({ type: 'error', gen, message: error instanceof Error ? error.message : String(error) })
  } finally {
    await reader.cancel().catch(() => {})
    reader.releaseLock()
    landmarker?.close()
    self.close()
  }
}

self.onmessage = (event: MessageEvent<TrackerRequest>): void => {
  if (event.data.type === 'stop') {
    stopped = true
    void reader?.cancel().catch(() => {})
  } else if (!reader) {
    void track(event.data.gen, event.data.readable)
  }
}
