from pathlib import Path

import pytest

from fovea.webcam.model import (
    FACE_LANDMARKER_SHA256,
    ModelChecksumError,
    download_face_landmarker,
    sha256_file,
    verify_face_landmarker,
)


def test_verify_rejects_corrupt_model(tmp_path: Path) -> None:
    path = tmp_path / "face_landmarker.task"
    path.write_bytes(b"not-a-real-model")
    with pytest.raises(ModelChecksumError, match="checksum mismatch"):
        verify_face_landmarker(path)


def test_download_fails_when_checksum_does_not_match(tmp_path: Path, monkeypatch) -> None:
    dest = tmp_path / "face_landmarker.task"

    def fake_retrieve(_url: str, filename: str) -> None:
        Path(filename).write_bytes(b"tampered")

    monkeypatch.setattr("fovea.webcam.model.urllib.request.urlretrieve", fake_retrieve)
    with pytest.raises(ModelChecksumError):
        download_face_landmarker(dest, expected=FACE_LANDMARKER_SHA256)
    assert not dest.exists()
    assert not list(tmp_path.glob("*.partial"))


def test_existing_file_is_rechecked(tmp_path: Path) -> None:
    path = tmp_path / "face_landmarker.task"
    path.write_bytes(b"abc")
    digest = sha256_file(path)
    verify_face_landmarker(path, expected=digest)
    download_face_landmarker(path, expected=digest)
    with pytest.raises(ModelChecksumError):
        verify_face_landmarker(path, expected="0" * 64)
