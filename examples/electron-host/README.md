# Electron host example

This intentionally small host keeps the camera indicator visible, registers a
DOM button for target-aware dwell, maps display-normalized gaze into the window,
and renders calibration cues itself.

From this directory, after the Python package and model are available:

```bash
npm install
FOVEA_BINARY=/absolute/path/to/fovea npm start
```

The local file dependency builds `packages/fovea-client` during packaging. This
example is source-only until the client package is published by a maintainer.
