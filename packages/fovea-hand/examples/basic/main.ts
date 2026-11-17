import { HandGesture, HandTargeting, type HandFrame } from '@wega-labs/fovea-hand'
import { HandCamera, DOMHandTargets, drawHandSkeleton } from '@wega-labs/fovea-hand/browser'

const status = document.querySelector<HTMLElement>('#status')!
const canvas = document.querySelector<HTMLCanvasElement>('#skeleton')!
const aim = document.querySelector<HTMLElement>('#aim')!
const progress = document.querySelector<HTMLElement>('#progress')!
const toggle = document.querySelector<HTMLButtonElement>('#camera')!
let camera: HandCamera | null = null
let gen = 0
let watchdog: ReturnType<typeof setInterval> | undefined
function stop() {
  camera?.stop()
  camera = null
  clearInterval(watchdog)
  toggle.textContent = 'Start camera'
  status.textContent = 'Camera off'
  aim.style.display = 'none'
  canvas.width = 560
}
function select(name: string) {
  status.textContent = `Selected ${name}`
}
document
  .querySelectorAll<HTMLButtonElement>('[data-hand]')
  .forEach((button) => (button.onclick = () => select(button.textContent!)))
toggle.onclick = () => {
  if (camera) {
    stop()
    return
  }
  const session = ++gen,
    gesture = new HandGesture(session),
    targeting = new HandTargeting(),
    dom = new DOMHandTargets()
  let lastAccepted = Date.now()
  function receive(frame: HandFrame) {
    const now = performance.timeOrigin + performance.now()
    const result = gesture.step(frame, now, innerWidth, innerHeight)
    if (!result.accepted) return
    lastAccepted = Date.now()
    drawHandSkeleton(canvas, frame)
    const sample = gesture.sample
    if (!sample) {
      targeting.reset()
      aim.style.display = 'none'
      return
    }
    const targets = dom.collect(document.querySelectorAll<HTMLButtonElement>('[data-hand]'), (button) => ({
      label: button.textContent!,
      activation: 'dwell'
    }))
    const feedback = targeting.update(sample.point, now, targets, gesture.phase === 'armed' && !sample.pinching)
    aim.style.display = feedback.target ? 'block' : 'none'
    if (feedback.target) {
      const r = feedback.target.rect
      Object.assign(aim.style, {
        left: `${r.x - 5}px`,
        top: `${r.y - 5}px`,
        width: `${r.width + 10}px`,
        height: `${r.height + 10}px`
      })
    }
    progress.style.width = `${feedback.progress * 100}%`
    if (feedback.activate && dom.resolve(feedback.activate.id)) select(feedback.activate.label)
  }
  camera = new HandCamera(session, {
    onFrame: receive,
    onReady: () => {
      status.textContent = 'Show an open hand, then point and pause'
    },
    onError: (message) => {
      stop()
      status.textContent = message
    }
  })
  toggle.textContent = 'Stop camera'
  status.textContent = 'Starting camera…'
  void camera.start()
  watchdog = setInterval(() => {
    if (Date.now() - lastAccepted > 400) {
      gesture.abort()
      targeting.reset()
      aim.style.display = 'none'
    }
    if (Date.now() - lastAccepted > 3000) {
      stop()
      status.textContent = 'Tracking stopped responding'
    }
  }, 100)
}
window.addEventListener('pagehide', stop)
document.addEventListener('visibilitychange', () => {
  if (document.hidden) stop()
})
