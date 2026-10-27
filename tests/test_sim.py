import ast
import inspect
import json
from dataclasses import replace

import numpy as np
import pytest

from lost_together import knowledge, planner
from lost_together.config import Cooperation, EpisodeConfig, Perception
from lost_together.rng import streams
from lost_together.sim import run_episode
from lost_together.world import distance_field, generate

FAST = replace(EpisodeConfig(), t_max=250)


def cfg_with(**kw):
    c = FAST
    if "coop" in kw:
        c = replace(c, cooperation=kw["coop"])
    if "mode" in kw:
        c = replace(c, perception=replace(c.perception, mode=kw["mode"]))
    if "predator_on" in kw:
        c = replace(c, predator_on=kw["predator_on"])
    return c


def test_same_seed_same_result():
    a = run_episode(cfg_with(coop=Cooperation.SHARE), 3)
    b = run_episode(cfg_with(coop=Cooperation.SHARE), 3)
    assert a == b


def test_world_is_identical_across_conditions():
    """Common random numbers: the forest depends only on the seed."""
    w1 = generate(FAST.world, streams(7)["world"])
    w2 = generate(FAST.world, streams(7)["world"])
    assert np.array_equal(w1.passable, w2.passable) and w1.exits == w2.exits and w1.trailhead == w2.trailhead


def test_world_is_connected_and_exits_on_edge():
    w = generate(FAST.world, streams(1)["world"])
    d = distance_field(np.where(w.passable, 1.0, np.inf), [w.trailhead])
    n = w.size
    for e in w.exits:
        assert np.isfinite(d[e])
        assert e[0] in (0, n - 1) or e[1] in (0, n - 1)


def test_perfect_perception_has_no_false_sightings():
    for s in range(3):
        r = run_episode(cfg_with(mode=Perception.PERFECT), s)
        assert r["n_sightings_false"] == 0


def test_predator_off_means_no_captures_and_no_alarms():
    for s in range(3):
        r = run_episode(cfg_with(predator_on=False, mode=Perception.PERFECT), s)
        assert r["n_caught"] == 0 and r["n_alarm_decisions"] == 0 and r["threat_onset"] == -1


@pytest.mark.parametrize("coop", list(Cooperation))
def test_metrics_are_in_range(coop):
    r = run_episode(cfg_with(coop=coop), 0)
    assert 0.0 <= r["frac_escaped"] <= 1.0
    assert r["n_caught"] + round(r["frac_escaped"] * r["n_hikers"]) <= r["n_hikers"]


def test_hiker_policy_never_touches_ground_truth():
    """No-leakage rule: the policy and map modules must not reference ground-truth objects."""
    forbidden = {"World", "Predator", "TruthLog", "generate", "world", "truth"}
    for mod in (planner, knowledge):
        tree = ast.parse(inspect.getsource(mod))
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        names |= {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names}
        names |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        assert not names & forbidden, f"{mod.__name__} references {names & forbidden}"


def test_group_survives_everyone_being_caught():
    """Regression: a fast, far-sensing predator can catch the last hiker before the group moves."""
    c = replace(EpisodeConfig(), cooperation=Cooperation.GROUP,
                predator=replace(EpisodeConfig().predator, speed_ratio=1.5, sense_radius=40))
    r = run_episode(c, 2)
    assert r["n_caught"] + round(r["frac_escaped"] * r["n_hikers"]) <= r["n_hikers"]


def test_calibrated_perception_uses_table(tmp_path):
    table = {
        "p_detect": {"0.25": {"0.0": {"5": 1.0, "60": 1.0}, "0.2": {"5": 0.0, "60": 0.0}}},
        "conf_quantiles": {"0.0": {"5": [0.6, 0.8, 0.9], "60": [0.6, 0.8, 0.9]},
                           "0.2": {"5": [0.6, 0.8, 0.9], "60": [0.6, 0.8, 0.9]}},
        "false_bear_per_frame": {"0.0": 0.0, "0.2": 0.0},
        "decoy_as_bear": {"0.0": 0.0, "0.2": 0.0},
    }
    path = tmp_path / "cal.json"
    path.write_text(json.dumps(table))
    base = replace(FAST, perception=replace(FAST.perception, mode=Perception.CALIBRATED, calibration=str(path)))
    clear = run_episode(replace(base, perception=replace(base.perception, beta=0.0)), 0)
    blind = run_episode(replace(base, perception=replace(base.perception, beta=0.2)), 0)
    assert clear["n_sightings_false"] == 0 and blind["n_sightings_real"] == 0
    assert clear["beta"] == 0.0 and blind["beta"] == 0.2


def test_distance_field_respects_walls():
    cost = np.ones((5, 5))
    cost[1:4, 2] = np.inf
    d = distance_field(cost, [(2, 0)])
    assert np.isinf(d[2, 2])
    assert d[2, 4] > d[0, 4] - 1e-9  # must route around the wall
