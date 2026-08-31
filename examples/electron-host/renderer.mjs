const gaze = document.querySelector("#gaze");
const cue = document.querySelector("#cue");
const status = document.querySelector("#status");
const tracking = document.querySelector("#tracking");
const calibrate = document.querySelector("#calibrate");
const targets = [...document.querySelectorAll("[data-fovea-target]")];

function registerTargets() {
  window.fovea.registerTargets(
    targets.map((element) => {
      const rectangle = element.getBoundingClientRect();
      return {
        id: element.dataset.foveaTarget,
        x: rectangle.x,
        y: rectangle.y,
        width: rectangle.width,
        height: rectangle.height,
      };
    }),
  );
}

window.addEventListener("resize", registerTargets);
window.addEventListener("load", registerTargets);
window.fovea.on("status", (message) => {
  status.textContent = message;
});
window.fovea.on("tracking", (value) => {
  tracking.textContent = value;
  document.body.dataset.tracking = value;
});
window.fovea.on("gaze", (point) => {
  gaze.hidden = point === null;
  if (point === null) return;
  gaze.style.transform = `translate(${point.x - 9}px, ${point.y - 9}px)`;
});
window.fovea.on("dwell-progress", ({ id, progress }) => {
  const target = document.querySelector(`[data-fovea-target="${CSS.escape(id)}"]`);
  target?.style.setProperty("--dwell", `${Math.round(progress * 100)}%`);
});
window.fovea.on("dwell", (id) => {
  document.querySelector(`[data-fovea-target="${CSS.escape(id)}"]`)?.click();
});
window.fovea.on("calibration-cue", ({ x, y }) => {
  cue.hidden = false;
  cue.style.transform = `translate(${x - 14}px, ${y - 14}px)`;
});

calibrate.addEventListener("click", async () => {
  calibrate.disabled = true;
  cue.hidden = false;
  try {
    const result = await window.fovea.calibrate();
    status.textContent = `calibrated · LOO error ${result.loo_error.toFixed(3)}`;
  } catch (error) {
    status.textContent = error.message;
  } finally {
    cue.hidden = true;
    calibrate.disabled = false;
    registerTargets();
  }
});

document.querySelector("#demo-action").addEventListener("click", () => {
  status.textContent = "Dwell activated the demo target";
});
