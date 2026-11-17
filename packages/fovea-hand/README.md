# `@wega-labs/fovea-hand`

A framework-independent hand input package: local GPU camera tracking, responsive smoothing, curl-click/drag/scale intents, forgiving target acquisition, dwell activation, relative palm placement, reach calibration, and skeleton drawing.

The package has **no host store, React, Electron, IPC, or backend dependency**. Hosts provide targets and decide what an activation does. It is licensed under Apache-2.0 (see `LICENSE`); the bundled MediaPipe model and WASM (Apache-2.0) and `1eurofilter` (BSD-3-Clause) ship with their own notices under `dist/LICENSES`. The npm package is not published yet; publishing under the Wega Labs organization is a maintainer action.

## Build and test

From this directory:

```sh
npm install        # also builds dist/ (prepare)
npm run check:core
npm run lint
npm test
npm run build
npm run smoke      # packs the tarball and proves it in a host outside this repository
npm pack
```

The tarball exports compiled ESM and TypeScript declarations. It includes the pinned MediaPipe model; the exact MediaPipe dependency supplies the WASM assets. Build output stays under `dist/`.

Every build runs `check:core`. It follows the core entry's complete TypeScript dependency
graph (including type imports and re-exports), compiles strictly with only ES2023 ambient
types and no automatic `@types` packages, and rejects browser modules, React, Electron,
host source, injected DOM libraries, and unresolved dynamic imports. The core's only
external dependency is `1eurofilter`. Browser code is checked separately with DOM types.
`test/core-boundary.test.mjs` proves the check rejects each of those escapes.

## Browser host

The [standalone example](examples/basic) runs without any application framework:

```sh
cd examples/basic
npm install
npm run dev
```

Use a secure context (`https://` or localhost), a Chromium browser with `MediaStreamTrackProcessor` on `window`, transferable streams, and worker WebGL2 support. Permission or GPU failure is an explicit error. There is no CPU delegate fallback or runtime CDN fetch. Other bundlers than Vite have not been validated.

Vite hosts need three settings, all shown in the example's `vite.config.ts`:

- `worker: { format: 'es' }`: MediaPipe's WASM loader is an ES module; the default IIFE worker cannot import it.
- `optimizeDeps: { exclude: ['@wega-labs/fovea-hand'], include: ['@wega-labs/fovea-hand > @mediapipe/tasks-vision', '@wega-labs/fovea-hand > 1eurofilter'] }`: dependency prebundling inlines the browser entry into one file and breaks its relative module-worker URL (404 in development). Excluding the package also stops Vite's dependency scanner at its boundary, so its own dependencies (`1eurofilter` is CommonJS and must be prebundled) are listed explicitly; otherwise their first use triggers a re-optimization and a full page reload.
- `server.fs.allow` including `realpathSync('node_modules/@wega-labs/fovea-hand')` when the package is linked (`file:` or `npm link`): its worker, WASM, and model are served through `/@fs/` from outside the Vite root and are otherwise rejected with 403. An ordinary installed copy inside `node_modules` needs no allow entry.

CSP needs `script-src 'self' 'wasm-unsafe-eval'` and local module workers.

```ts
import { HandGesture, HandTargeting } from '@wega-labs/fovea-hand'
import { HandCamera, DOMHandTargets } from '@wega-labs/fovea-hand/browser'

const gen = 1 // increment for each camera session
const gesture = new HandGesture(gen)
const targeting = new HandTargeting({ dwellMs: 700, radius: 28, strength: 0.65 })
const dom = new DOMHandTargets()
const camera = new HandCamera(gen, {
  onReady: () => showStatus('Point and pause'),
  onError: showError,
  onFrame(frame) {
    const now = performance.timeOrigin + performance.now()
    if (!gesture.step(frame, now, innerWidth, innerHeight).accepted) return
    const sample = gesture.sample
    if (!sample) {
      targeting.reset()
      return
    }
    const targets = dom.collect(document.querySelectorAll('[data-hand]'), (element) => ({
      label: element.getAttribute('aria-label') || element.textContent!,
      activation: 'dwell'
    }))
    const result = targeting.update(sample.point, now, targets, gesture.phase === 'armed' && !sample.pinching)
    renderHighlight(result)
    if (result.activate && dom.resolve(result.activate.id)) {
      performHostAction(result.activate.id)
    }
  }
})
await camera.start()
// camera.stop() on disable, teardown, or error; safe even while permission is pending.
```

`showStatus`, `showError`, `renderHighlight`, and `performHostAction` belong to the host. The complete example also handles missing-frame timeouts and page lifecycle.

## Public API and responsibilities

- **Core entry (`@wega-labs/fovea-hand`)**: `HandGesture`, `HandPanGesture`, `HandSnapGesture`, `isTwoFingerPanPose`, `isOpenHandPose`, frame/point/region types, `HandTargeting`, `HandTarget`/`TargetFeedback`, `HoldToPlace`, `HandReachCalibration`, mapping and viewport math. Importing the core does not access DOM globals or load MediaPipe.
- **Browser entry (`@wega-labs/fovea-hand/browser`)**: `HandCamera`, `DOMHandTargets`, `drawHandSkeleton`, camera error formatting, worker protocol types.
- Frame times are epoch milliseconds (`wallTs`); detector `ts` is monotonic milliseconds. `HAND_FRAME_MAX_AGE_MS` is 150: older results, future timestamps, repeated timestamps, and old generations are rejected. `HAND_TRACKING_LOSS_MS` is 400: longer gaps cancel gestures. A fresh sample after a shorter stall rebases an established manipulation without replaying missed travel. Accepted input gaps reset dwell; the host must also cancel held actions when frames stop arriving.
- Target rectangles and pointer coordinates are in viewport CSS pixels. The host supplies only visible, currently actionable targets. IDs must change if an action's meaning changes. `DOMHandTargets.capture(element, policy)` revalidates one control without rediscovering the document, using visibility/occlusion checks and current policy. A policy's optional `key` identifies actions whose labels can remain unchanged while their meaning changes.
- Targets explicitly opt into `activation: 'dwell'`. `manual` targets, such as sliders and drag handles, can be highlighted but never activate from a dwell. Ambiguous peers are returned without guessing.
- Activation blocks re-entry under the same held aim, including a replacement control appearing in the same place. The hand must leave the old target's neighborhood. Changing targets, target movement, insufficient stability, and tracking gaps restart the dwell timer.
- `HoldToPlace` starts from a palm position. It requires actual displacement before a stable pause can finish. Hosts may apply its relative delta to a drag, preview, viewport, or other manipulation.
- `HandTargeting` accepts `radius` (0–100 CSS pixels) and `strength` (0–1). Either zero disables magnetic acquisition/attraction outside the target. Direct dwell remains available. `feedback.point` applies spatial attraction toward the nearest inner edge, preserving precision within large targets. Use raw pointing coordinates for dwell stability, and revalidate the target before activating it.
- `HandTargeting.searchRadius` includes the current target's retention margin. Hosts may
  use it to reject distant rectangles before costly visibility/hit tests. Refresh this
  nearby set on each pointing sample, and retain any separately offered action choices
  until they are dismissed. Browser discovery remains independent of the core.
- `HandPanGesture(gen).step(frame, now)` emits `start`, `move`, `end`, and `cancel` pan intents with time-integrated camera-normalized `delta` (X mirrored, Y downward). A small offset from the initial palm position controls continuous speed; holding that offset keeps traveling. Returning to neutral stops immediately, including any filter drift. Speed reaches its cap at 0.08 camera widths; movement only advances on fresh frames. Multiply by viewport size and a fixed host sensitivity independent of the pointing region. Reserve input during both `arming` and `panning`, including the ending frame; start has zero delta. Acquisition requires one visible hand with index/middle extended, ring/little folded, and no pinch. Established strokes accept relaxed angles and briefly tolerate ambiguity; explicit folding or pinching ends motion. Use `isOpenHandPose` to return to pointing. Hosts must suppress pointer/dwell while repositioning, arbitrate with active drags, and abort on tracking loss or context changes.
- `HandSnapGesture(gen).step(frame, now)` returns `accepted`, `active`, and `triggered`. Reserve pointer/dwell input whenever `active`, including the triggering frame. Recognition uses thumb/middle contact followed by a quick inward middle-finger flick and confirmed separation. It accepts a prepared hold or a single contact frame, follows either wrist beside a passive hand, and debounces repeat snaps. Index pinches, tracking gaps and hand jumps cancel the sequence. Call `abort()` on context changes and arbitrate against other active gestures. The host decides the action.
- `HandGesture(gen, region, interaction)` defaults to `'pinch'`: a stationary pinch emits one `down`/`up` pair on release, while palm travel starts a direct drag. Passing `'curl'` opts into the retained experiment; curl recognition and its finger-metric work are disabled in the default mode. The host selects the mode.
- In `'curl'` mode, `HandGesture` emits a distinct `click` on a confirmed index curl followed by straightening. `curling` begins as the finger bends: hosts must latch the previous highlighted target, freeze pointing/dwell, and revalidate that target before activation. Curl and release each need 60 ms; a shallow bend or one-frame spike does not click. The thumb stays separate; other fingers can stay open or folded as they were during pointing. Tracking interruption, a second hand, closing the whole open hand, a hold over 1.5 seconds, or wrist departure over 0.12 camera widths (aspect-corrected) cancels the click. Small palm movement is allowed and never becomes a drag. Curl geometry shares the pan recognizer's finger metrics.
- In `'curl'` mode, stationary pinches do not emit a click or a down/up pair. The host decides what can be dragged.
- A second confirmed index/thumb pinch starts scaling from either a pending pinch or an active drag. A drag-to-scale transition emits `cancel` before `scale:start`; a native host must drain the cancelled press before acquiring a resize handle, retain only the latest span while waiting, and abandon the deferred start on release, loss, or context changes. Scale anchors are unbounded and scale travel is independent of the pointing region.
- Pinch drag/scale uses palm displacement after preserving the initial pointing anchors. Close confirmation (40 ms) overlaps with the 4 CSS pixel palm travel requirement; release requires relaxed separation (thumb/index distance above 0.55 palm lengths) for 60 ms. Pointer mapping clamps to the active region; relative palm mapping remains unbounded so small reach regions cannot stall manipulation.
- `HandReachCalibration` records only normalized fingertip coordinates. Collect six seconds of accepted open-hand samples, call `finish()`, then create a new gesture with the returned region. Null means insufficient movement or samples.
- The package emits intent; it does not synthesize trusted browser or OS input. Browser hosts can invoke their own actions. Electron hosts can provide a validated native input adapter. Never represent `dispatchEvent` as trusted input.
- Hosts own camera status, focus/visibility policy, physical input takeover, cancellation, undo, accessibility labels, and persistence. Call `gesture.abort()` and cancel the corresponding host action together. Stopping a camera alone cannot release a host's held mouse button.

## Writing a host adapter

A host imports only the two public entries. Its adapter supplies target policies (which controls are dwell-activatable, which are manual), maps activations and drag/scale/pan intents onto its own actions, and owns any trusted input transport. Keep the adapter, its UI, and its persistence in the host; the package never learns about them. The camera and model assets live in this package's `dist/`, not in the host's source tree.
