MediaPipe FaceLandmarker model files (``*.task``) belong in this directory.

The download is **pinned** to Face Landmarker float16 **revision 1** (not
``latest``) and verified with SHA-256 before use.

| Field | Value |
| --- | --- |
| Version | `1` |
| URL | `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task` |
| SHA-256 | `64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff` |

```bash
python scripts/download_mediapipe_model.py
```

Constants live in `src/fovea/webcam/model.py`
(`FACE_LANDMARKER_VERSION`, `FACE_LANDMARKER_URL`, `FACE_LANDMARKER_SHA256`).
Changing the model requires updating the version, URL, and checksum together.
A checksum mismatch is a hard error; the file is not used.
