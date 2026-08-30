"""Calibration target layout and on-screen rendering."""

from fovea.webcam.calibration import CALIBRATION_LAYOUT, CalibrationTarget
from fovea.webcam.calibration_view import render_calibration_frame
from fovea.webcam.engine import WizardState


def test_pixel_mapping_scales_with_resolution() -> None:
    target = CalibrationTarget("top_right", 0.88, 0.12)
    x_hd, y_hd = target.pixel_xy(1920, 1080)
    x_sd, y_sd = target.pixel_xy(640, 480)
    assert x_hd == round(0.88 * 1919)
    assert y_hd == round(0.12 * 1079)
    assert x_sd == round(0.88 * 639)
    assert y_sd == round(0.12 * 479)


def test_layout_is_the_single_source_of_calibration_positions() -> None:
    assert len(CALIBRATION_LAYOUT) == 10
    labels = [t.label for t in CALIBRATION_LAYOUT]
    assert labels[0] == "center"
    assert labels[-1] == "bottom_right"


def test_render_highlights_active_layout_target() -> None:
    active = CALIBRATION_LAYOUT[3]
    wizard = WizardState(
        kind="calibrate",
        index=3,
        label=active.label,
        sx=active.x,
        sy=active.y,
        samples=4,
        needed=28,
        quality="FAIR",
        instruction="Look at the point.",
    )
    for width, height in ((640, 480), (1920, 1080)):
        frame = render_calibration_frame(width, height, wizard)
        assert frame.shape == (height, width, 3)
        px, py = active.pixel_xy(width, height)
        pixel = frame[py, px]
        assert int(pixel[0]) + int(pixel[1]) + int(pixel[2]) > 200
        idle = CALIBRATION_LAYOUT[1]
        ix, iy = idle.pixel_xy(width, height)
        idle_pixel = frame[iy, ix]
        assert int(idle_pixel.sum()) < int(pixel.sum())
