"""Fast, offline tests for lost_together.vision (composite geometry, fog, SGBM units, PnP odometry)."""
import cv2
import numpy as np
import pytest

from lost_together.vision import composite as C
from lost_together.vision import odometry as O
from lost_together.vision.stereo import StereoDepth, disparity_to_depth

H_CAM = 1.2


def ground_frame(tree_cols=None, tree_depth=5.0):
    """Synthetic 640x640 frame: flat ground plane (seg = ground) below the horizon, sky above."""
    h = w = 640
    v = np.arange(h, dtype=np.float32)[:, None].repeat(w, 1)
    depth = np.where(v > C.CY + 0.5, C.F * H_CAM / np.maximum(v - C.CY, 1e-3), 1.5e4).astype(np.float32)
    seg = np.where(v > C.CY + 0.5, 64, 27).astype(np.uint8)
    rng = np.random.default_rng(0)
    left = rng.integers(60, 120, (h, w, 3), dtype=np.uint8)
    if tree_cols is not None:
        depth[:, tree_cols[0]:tree_cols[1]] = tree_depth
        seg[:, tree_cols[0]:tree_cols[1]] = 199
    return C.Frame(left, left.copy(), depth, seg, "synthetic")


@pytest.fixture(autouse=True)
def fake_cutout(monkeypatch):
    """A 200x100 opaque bright-red 'bear' so tests need no downloaded photos."""
    bgra = np.zeros((100, 200, 4), np.uint8)
    bgra[..., 2], bgra[..., 3] = 230, 255
    monkeypatch.setattr(C, "load", lambda cid: bgra)
    monkeypatch.setattr(C, "cls_of", lambda cid: "bear")


def test_pixel_height_and_foot_row_follow_pinhole_geometry():
    fr = ground_frame()
    for z in (5.0, 10.0, 20.0):
        r = C.render(fr, [("x", z, 0.0)], beta=0.0, harmonize=0)
        x, y, w, h = r["boxes"][0]
        assert h == round(C.F * 1.0 / z)                      # bear H = 1.0 m
        assert w == round(200 * h / 100)
        foot = r["placements"][0]["foot_row"]
        assert abs(foot - (C.CY + C.F * H_CAM / z)) <= 2.0    # ground row where GT depth == Z
        assert r["placements"][0]["foot_method"].startswith("seg+depth")
        assert abs((x + w / 2) - C.CX) <= 1                   # bearing 0 -> principal column
        assert r["visible"][0] > 0.95


def test_bearing_maps_to_column():
    assert C.bearing_to_col(0) == pytest.approx(320)
    assert C.bearing_to_col(45) == pytest.approx(640)
    assert C.bearing_to_col(-20) == pytest.approx(320 - 320 * np.tan(np.radians(20)))


def test_nearer_scene_occludes_object_per_pixel():
    fr = ground_frame(tree_cols=(310, 330), tree_depth=5.0)   # a trunk at 5 m in front of a bear at 10 m
    r = C.render(fr, [("x", 10.0, 0.0)], beta=0.0, harmonize=0)
    m = r["masks"][0]
    assert m[:, 310:330].max() == 0                          # hidden behind the trunk
    assert m[:, 290:305].max() > 0.9                         # visible either side
    assert 0.5 < r["visible"][0] < 0.95
    assert np.all(r["depth"][m > 0.5] == pytest.approx(10.0))
    # an object in front of the trunk is not occluded
    r2 = C.render(fr, [("x", 4.0, 0.0)], beta=0.0, harmonize=0)
    assert r2["masks"][0][:, 310:330].max() > 0.9


def test_right_image_shifted_by_disparity():
    fr = ground_frame()
    z = 10.0
    r = C.render(fr, [("x", z, 0.0)], beta=0.0, harmonize=0)
    cl = np.nonzero(r["masks"][0].sum(0))[0].mean()
    cr = np.nonzero(r["masks_right"][0].sum(0))[0].mean()
    assert cl - cr == pytest.approx(C.F * C.BASELINE / z, abs=1.0)


def test_fog_is_identity_at_zero_and_monotone_in_beta_and_depth():
    img = np.full((4, 4, 3), 40, np.uint8)
    d = np.array([[1, 10, 100, 1e4]] * 4, np.float32)
    assert np.array_equal(C.fog(img, d, 0.0), img)
    prev = img.astype(int)
    for beta in (0.02, 0.05, 0.1, 0.2):
        f = C.fog(img, d, beta).astype(int)
        assert np.all(f >= prev)                       # darker-than-airlight pixels brighten as beta grows
        assert np.all(np.diff(f[0, :, 0]) >= 0)        # and with depth
        prev = f
    assert C.fog(img, d, 0.2)[0, 3, 0] == pytest.approx(C.AIRLIGHT, abs=1)
    # object contrast against the background drops monotonically with fog
    fr = ground_frame()
    contrasts = []
    for beta in (0.0, 0.05, 0.1, 0.2):
        r = C.render(fr, [("x", 20.0, 0.0)], beta=beta, harmonize=0)
        m = r["masks"][0] > 0.5
        contrasts.append(abs(float(r["left"][m][:, 2].mean()) - float(r["left"][~m][:, 2].mean())))
    assert all(a > b for a, b in zip(contrasts, contrasts[1:]))


def test_sgbm_fixed_point_disparity_to_depth():
    disp16 = np.array([[16 * 8, 16 * 1, 0, -16, 16 * 32]], np.int16)
    d = disparity_to_depth(disp16)
    assert d[0, 0] == pytest.approx(C.F * C.BASELINE / 8)    # 10 m
    assert d[0, 1] == 0 and d[0, 2] == 0 and d[0, 3] == 0    # disp <= 1 px masked
    assert d[0, 4] == pytest.approx(2.5)


def test_sgbm_recovers_depth_of_shifted_texture():
    rng = np.random.default_rng(1)
    tex = cv2.GaussianBlur(rng.integers(0, 255, (480, 800), dtype=np.uint8), (0, 0), 1.2)
    shift = 16                                                # px disparity -> 320*0.25/16 = 5 m
    left = np.ascontiguousarray(tex[:, 0:640])
    right = np.ascontiguousarray(tex[:, shift:shift + 640])
    depth, _ = StereoDepth().compute(cv2.cvtColor(left, cv2.COLOR_GRAY2BGR), cv2.cvtColor(right, cv2.COLOR_GRAY2BGR))
    valid = depth[:, 160:600] > 0
    assert valid.mean() > 0.9
    assert np.median(depth[:, 160:600][valid]) == pytest.approx(5.0, rel=0.02)


def _plane_view(tex, z, tx):
    """Fronto-parallel textured plane at depth z seen by a camera translated by tx (m) along x."""
    h = w = 480
    u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    X = (u - C.CX) * z / C.F + tx
    Y = (v - C.CY) * z / C.F
    img = cv2.remap(tex, X * 120 + tex.shape[1] / 2, Y * 120 + tex.shape[0] / 2, cv2.INTER_LINEAR)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), np.full((h, w), z, np.float32)


def test_pnp_odometry_recovers_known_shift():
    rng = np.random.default_rng(2)
    tex = cv2.GaussianBlur(rng.integers(0, 255, (1600, 1600), dtype=np.uint8), (0, 0), 2.0)
    tex = cv2.normalize(tex, None, 0, 255, cv2.NORM_MINMAX)
    a, da = _plane_view(tex, 6.0, 0.0)
    b, db = _plane_view(tex, 6.0, 0.3)
    for mode in ("pnp", "gated"):
        poses, log = O.run([a, b], [da, db], mode=mode)
        assert poses[1][:3, 3] == pytest.approx([0.3, 0.0, 0.0], abs=0.03), (mode, log)
        assert O.rot_deg(poses[1][:3, :3]) < 0.5
        assert log[0]["src"] in ("pnp", "dense", "pnp_weak")


@pytest.mark.skipif(not O.POSES.exists(), reason="TartanGround poses not downloaded")
def test_gt_pose_conversion_is_orthonormal_optical_frame():
    T = O.load_gt_poses(frames=range(0, 3))
    for M in T:
        R = M[:3, :3]
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-6) and np.linalg.det(R) == pytest.approx(1.0)
    # optical z (forward) == body x: identity body rotation -> optical z axis = world x
    assert np.allclose(O.quat_to_R(0, 0, 0, 1) @ O.R_BODY_OPT @ [0, 0, 1], [1, 0, 0])


@pytest.mark.skipif(not (C.TG / "ForestEnv_P0004" / "depth_lcam_front" / "000001_lcam_front_depth.png").exists(),
                    reason="TartanGround P0004 frames not downloaded")
def test_dense_icp_needs_rescaled_depth():
    """OpenCV 5.0.0 Odometry returns ~0 translation on metric forest depth; at DENSE_SCALE units ICP recovers
    the GT step (P0004 0->1 is a straight 0.42 m step)."""
    gtw = O.load_gt_poses(frames=range(0, 2))
    T_gt = np.linalg.inv(gtw[1]) @ gtw[0]
    a, b = C.load_frame("P0004", 0), C.load_frame("P0004", 1)
    d0 = np.where(a.depth < O.MAX_DEPTH, a.depth, 0).astype(np.float32)
    d1 = np.where(b.depth < O.MAX_DEPTH, b.depth, 0).astype(np.float32)
    g0, g1 = (cv2.cvtColor(f.left, cv2.COLOR_BGR2GRAY) for f in (a, b))
    metric = O.dense_estimate(O.make_odometry(kind="DEPTH", scale=1.0), d0, g0, d1, g1, "DEPTH", scale=1.0)
    assert np.linalg.norm(metric[:3, 3]) < 0.01                       # the 5.0.0 failure mode
    T = O.dense_estimate(O.make_odometry(kind="DEPTH"), d0, g0, d1, g1, "DEPTH")
    assert np.linalg.norm((np.linalg.inv(T_gt) @ T)[:3, 3]) < 0.02


def test_right_depth_warp_fills_border_band_with_scene_depth():
    """The right-border band the left camera never sees must not be filled as sky (it would fog to white)."""
    d = np.full((20, 640), 2.0, np.float32)                  # 2 m wall -> 40 px disparity
    r = C.warp_depth_to_right(d)
    assert np.allclose(r, 2.0)
    assert C.warp_depth_to_right(np.full((4, 64), 1e4, np.float32)).max() == pytest.approx(C.SKY_CLIP)
