import json

import numpy as np

from lost_together.config import Cooperation
from lost_together.export import (NEVER, decode_mask, decode_u16, encode_episode, encode_mask, export,
                                  make_config, write_index)
from lost_together.rng import streams
from lost_together.sim import run_episode
from lost_together.world import generate

STATUSES = {"active", "escaped", "caught"}
MODES = {"enjoy", "escape", "home"}


def test_mask_round_trip():
    rng = np.random.default_rng(0)
    for n in (5, 8, 80):
        m = rng.random((n, n)) < 0.4
        assert np.array_equal(decode_mask(encode_mask(m), n), m)


def test_export_schema_round_trip(tmp_path):
    entries = export(3, ["solo", "share"], tmp_path, t_max=120, verbose=False)
    index = json.loads((tmp_path / "index.json").read_text())
    assert [e["id"] for e in index["episodes"]] == [e["id"] for e in entries]
    assert index["default"] == {"left": entries[0]["id"], "right": entries[1]["id"]}

    for entry in index["episodes"]:
        ep = json.loads((tmp_path / entry["file"]).read_text())      # strict JSON (no NaN)
        cfg = make_config(ep["cooperation"], 1.0, 20.0, t_max=120)
        assert ep["id"] == entry["id"] and ep["schema"] == 1
        assert ep["config"]["t_max"] == 120 and ep["config"]["sense_radius"] == 20.0

        # world decodes back to the generated forest
        world = generate(cfg.world, streams(3)["world"])
        w = ep["world"]
        assert np.array_equal(decode_mask(w["passable"], w["size"]), world.passable)
        assert [tuple(e) for e in w["exits"]] == world.exits and tuple(w["trailhead"]) == world.trailhead

        # metrics match a fresh untraced run (same seed, same config)
        row = run_episode(cfg, 3)
        for k in ("frac_escaped", "n_caught", "t_end", "n_relooks", "n_sightings_real"):
            assert ep["metrics"][k] == row[k]
        assert "frames" not in ep["metrics"] and "world" not in ep["metrics"]

        frames = ep["frames"]
        ts = [f["t"] for f in frames]
        assert ts == sorted(ts) and ts[0] == 0
        n_cells = w["size"] ** 2
        seen = set()
        for f in frames:
            assert len(f["h"]) == ep["config"]["n_hikers"]
            for r, c, status, mode in f["h"]:
                assert 0 <= r < w["size"] and 0 <= c < w["size"]
                assert status in STATUSES and mode in MODES
            if f["p"] is not None:
                assert f["p"][2] in {"patrol", "pursue", "handling"}
            for sid, r, c, conf, h, real in f["s"]:
                assert 0 <= conf <= 1 and isinstance(real, bool)
            for e in f["e"]:
                assert e[1] in {"relook", "escape", "home", "escaped", "caught", "predator"}
            assert all(0 <= i < n_cells and i not in seen for i in f["n"])
            seen.update(f["n"])

        # the trace ends in the same final statuses as the metrics
        final = [s for s in (p.split(":")[1] for p in ep["metrics"]["t_end"].split(";"))]
        last = [h[2][0] for h in frames[-1]["h"]]
        assert last == final
        # every status change appears exactly once as an event
        n_events = sum(1 for f in frames for e in f["e"] if e[1] in ("escaped", "caught"))
        assert n_events == sum(s != "a" for s in final)


def test_encode_episode_is_deterministic():
    cfg = make_config(Cooperation.SHARE.value, 1.5, 20.0, t_max=80)
    a = encode_episode(run_episode(cfg, 5, trace=True), cfg)
    b = encode_episode(run_episode(cfg, 5, trace=True), cfg)
    assert json.dumps(a) == json.dumps(b)


def test_index_merges_and_keeps_existing(tmp_path):
    export(1, ["solo"], tmp_path, t_max=40, verbose=False)
    export(2, ["share"], tmp_path, t_max=40, verbose=False)
    ids = [e["id"] for e in json.loads((tmp_path / "index.json").read_text())["episodes"]]
    assert len(ids) == 2 and ids[0].startswith("seed1-solo") and ids[1].startswith("seed2-share")
    (tmp_path / f"{ids[0]}.json").unlink()                      # vanished files drop out of the index
    index = write_index(tmp_path, [], None)
    assert [e["id"] for e in index["episodes"]] == ids[1:]


def test_own_maps_rebuild_the_traced_union():
    """SOLO/SIGHTINGS export each hiker's own map; together they must equal the union sim.py traced."""
    for coop, sr in (("solo", 1.0), ("sightings", 1.5), ("solo", 2.0)):
        cfg = make_config(coop, sr, 20.0, t_max=200)
        ep = encode_episode(run_episode(cfg, 4, trace=True), cfg)
        w = ep["world"]
        n2 = w["size"] ** 2
        own = np.stack([decode_u16(b) for b in w["own_seen"]])
        assert own.shape == (ep["config"]["n_hikers"], n2) and ep["config"]["private_map"]
        union = np.full(n2, NEVER)
        for f in ep["frames"]:
            union[f["n"]] = f["t"]
        assert np.array_equal(own.min(axis=0), union)
        assert (own != NEVER).sum(axis=1).min() > 0          # every hiker saw something of its own
    cfg = make_config("share", 1.0, 20.0, t_max=60)
    ep = encode_episode(run_episode(cfg, 4, trace=True), cfg)
    assert "own_seen" not in ep["world"] and not ep["config"]["private_map"]
