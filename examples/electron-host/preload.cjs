const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("fovea", {
  calibrate: () => ipcRenderer.invoke("fovea:calibrate"),
  registerTargets: (rectangles) => ipcRenderer.send("fovea:targets", rectangles),
  on: (name, listener) => {
    const allowed = new Set([
      "status",
      "tracking",
      "gaze",
      "dwell-progress",
      "dwell",
      "calibration-cue",
    ]);
    if (!allowed.has(name)) throw new Error(`unsupported Fovea channel: ${name}`);
    ipcRenderer.on(`fovea:${name}`, (_event, value) => listener(value));
  },
});
