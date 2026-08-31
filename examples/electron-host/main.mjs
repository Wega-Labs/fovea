import { join } from "node:path";
import { fileURLToPath } from "node:url";

import { app, BrowserWindow, ipcMain, screen } from "electron";
import {
  isOffWindow,
  spawnFovea,
  toClientPoint,
} from "@wega-labs/fovea-client";

const directory = fileURLToPath(new URL(".", import.meta.url));
let window;
let fovea;
let targetRectangles = [];

function geometry() {
  const content = window.getContentBounds();
  const display = screen.getDisplayMatching(window.getBounds()).bounds;
  return { content, display };
}

function normalizedPoint(clientX, clientY) {
  const { content, display } = geometry();
  return {
    x: (content.x + clientX - display.x) / display.width,
    y: (content.y + clientY - display.y) / display.height,
  };
}

function calibrationTargets() {
  const { width, height } = window.getContentBounds();
  return [
    ["top-left", 0.12, 0.14],
    ["top-right", 0.88, 0.14],
    ["center", 0.5, 0.5],
    ["bottom-left", 0.12, 0.86],
    ["bottom-right", 0.88, 0.86],
  ].map(([label, x, y]) => ({
    label,
    ...normalizedPoint(Number(x) * width, Number(y) * height),
  }));
}

function applyTargets() {
  if (fovea.status !== "running") return;
  const { content, display } = geometry();
  fovea.setTargets(
    targetRectangles.map((rectangle) => ({
      id: rectangle.id,
      x: (content.x + rectangle.x - display.x) / display.width,
      y: (content.y + rectangle.y - display.y) / display.height,
      w: rectangle.width / display.width,
      h: rectangle.height / display.height,
    })),
  );
}

function startFovea() {
  fovea = spawnFovea({
    binary: process.env.FOVEA_BINARY || "fovea",
    args: ["--no-display"],
    reconnect: true,
    maxGazeHz: 30,
  });
  fovea.on("hello", (hello) => {
    window.webContents.send("fovea:status", `ready · ${hello.backend}`);
    applyTargets();
  });
  fovea.on("reconnecting", ({ attempt }) => {
    window.webContents.send("fovea:status", `reconnecting · attempt ${attempt}`);
  });
  fovea.on("processError", (error) => {
    window.webContents.send("fovea:status", error.message);
  });
  fovea.on("tracking_state", (event) => {
    window.webContents.send("fovea:tracking", event.status);
  });
  fovea.on("gaze_point", (event) => {
    const { display, content } = geometry();
    if (isOffWindow(event, display, content)) {
      window.webContents.send("fovea:gaze", null);
      return;
    }
    window.webContents.send("fovea:gaze", toClientPoint(event, display, content));
  });
  fovea.on("dwell_progress", (event) => {
    window.webContents.send("fovea:dwell-progress", event);
  });
  fovea.on("dwell", (event) => {
    window.webContents.send("fovea:dwell", event.id);
  });
}

app.whenReady().then(() => {
  window = new BrowserWindow({
    width: 900,
    height: 620,
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false,
      preload: join(directory, "preload.cjs"),
    },
  });
  window.loadFile(join(directory, "index.html"));
  startFovea();

  ipcMain.on("fovea:targets", (_event, rectangles) => {
    targetRectangles = rectangles;
    applyTargets();
  });
  ipcMain.handle("fovea:calibrate", async () => {
    return fovea.runCalibration({
      targets: calibrationTargets(),
      onCue: (cue) => {
        const { display, content } = geometry();
        window.webContents.send("fovea:calibration-cue", {
          ...cue,
          ...toClientPoint(cue, display, content),
        });
      },
    });
  });
});

app.on("before-quit", () => fovea?.close());
app.on("window-all-closed", () => app.quit());
