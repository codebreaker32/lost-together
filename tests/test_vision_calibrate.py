"""Offline tests for the calibration helpers (no detector, no downloaded data)."""
import numpy as np
import pytest

from lost_together.vision import calibrate as K


def test_iou_and_mask_box():
    assert K.iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert K.iou((0, 0, 10, 10), (5, 0, 10, 10)) == pytest.approx(50 / 150)
    assert K.iou((0, 0, 10, 10), (20, 20, 5, 5)) == 0.0
    m = np.zeros((50, 50), np.float32)
    m[10:20, 5:30] = 1
    assert K.mask_box(m) == (5.0, 10.0, 25.0, 10.0)
    assert K.mask_box(np.zeros((5, 5))) is None


def test_zoom_tile_maps_boxes_consistently():
    img = np.zeros((640, 640, 3), np.uint8)
    img[400:410, 300:320] = 255
    tile, (x0, y0) = K.zoom_tile(img, 310, 410, 3)
    assert tile.shape == img.shape
    ys, xs = np.nonzero(tile[..., 0] > 127)
    assert xs.min() == pytest.approx((300 - x0) * 3, abs=2) and ys.min() == pytest.approx((400 - y0) * 3, abs=2)


def test_surrogate_fit_recovers_logistic_parameters():
    rng = np.random.default_rng(0)
    a, b, c = 9.0, -3.5, -0.8
    recs = []
    for z in K.DISTANCES:
        for beta in K.BETAS:
            p = 1 / (1 + np.exp(-(a + b * np.log(z) + c * beta * z)))
            recs += [{"z": z, "beta": beta, "score": 0.9 if rng.random() < p else 0.0} for _ in range(400)]
    fit = K.fit_surrogate(recs, 0.25)
    assert fit["a"] == pytest.approx(a, rel=0.15)
    assert fit["b"] == pytest.approx(b, rel=0.15)
    assert fit["c"] == pytest.approx(c, rel=0.25)
    assert fit["cell_max_abs_err"] < 0.1


def test_wilson_interval_brackets_proportion():
    lo, hi = K._wilson(5, 20)
    assert lo < 0.25 < hi
    assert K._wilson(0, 0) == (None, None)
