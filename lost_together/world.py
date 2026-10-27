"""The forest: obstacles, exits, trailhead, line of sight and shortest-path fields.

Everything here is ground truth. Only the simulator and the evaluator may hold a World;
hikers see it solely through perception (see perception.py).
"""
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from .config import WorldConfig

NEIGHBOURS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


@dataclass
class World:
    passable: np.ndarray            # bool [size, size]
    exits: list[tuple[int, int]]
    trailhead: tuple[int, int]

    @property
    def size(self) -> int:
        return self.passable.shape[0]

    def is_exit(self, cell: tuple[int, int]) -> bool:
        return cell in self.exits


def generate(cfg: WorldConfig, rng: np.random.Generator) -> World:
    n = cfg.size
    noise = rng.random((n, n))
    for _ in range(cfg.smoothing_passes):
        noise = ndimage.uniform_filter(noise, size=5, mode="reflect")
    passable = noise > np.quantile(noise, cfg.obstacle_fraction)

    centre = np.argwhere(passable[n // 4: 3 * n // 4, n // 4: 3 * n // 4]) + n // 4
    trailhead = tuple(int(v) for v in centre[rng.integers(len(centre))])

    # keep only the region reachable from the trailhead
    labels, _ = ndimage.label(passable, structure=np.ones((3, 3)))
    passable = labels == labels[trailhead]

    border = np.zeros_like(passable)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
    candidates = np.argwhere(passable & border)
    exits = _spread_out(candidates, cfg.n_exits, rng)
    return World(passable=passable, exits=exits, trailhead=trailhead)


def _spread_out(cands: np.ndarray, k: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    """Farthest-point sampling so exits sit on different sides of the forest."""
    chosen = [cands[rng.integers(len(cands))]]
    while len(chosen) < k:
        d = np.min([np.hypot(*(cands - c).T) for c in chosen], axis=0)
        chosen.append(cands[int(np.argmax(d))])
    return [tuple(int(v) for v in c) for c in chosen]


def line_of_sight(passable: np.ndarray, a: tuple[int, int], b: tuple[int, int]) -> bool:
    steps = int(max(abs(b[0] - a[0]), abs(b[1] - a[1])))
    if steps <= 1:
        return True
    rr = np.rint(np.linspace(a[0], b[0], steps + 1)[1:-1]).astype(int)
    cc = np.rint(np.linspace(a[1], b[1], steps + 1)[1:-1]).astype(int)
    return bool(passable[rr, cc].all())


WALL = 1e9   # finite stand-in for impassable edges; anything beyond it is reported as inf


@lru_cache(maxsize=4)
def _graph(n: int):
    """Fixed 8-neighbour CSR structure for an n x n grid; only edge weights change per query."""
    idx = np.arange(n * n).reshape(n, n)
    us, vs, lens = [], [], []
    for dr, dc in NEIGHBOURS:
        r0, r1 = max(0, -dr), n - max(0, dr)
        c0, c1 = max(0, -dc), n - max(0, dc)
        us.append(idx[r0:r1, c0:c1].ravel())
        vs.append(idx[r0 + dr:r1 + dr, c0 + dc:c1 + dc].ravel())
        lens.append(np.full(us[-1].size, np.hypot(dr, dc)))
    u, v, length = np.concatenate(us), np.concatenate(vs), np.concatenate(lens)
    order = np.lexsort((v, u))
    u, v, length = u[order], v[order], length[order]
    indptr = np.searchsorted(u, np.arange(n * n + 1)).astype(np.int32)
    return u, v.astype(np.int32), length, indptr


def distance_field(cost: np.ndarray, goals) -> np.ndarray:
    """Cost-weighted shortest distance from every cell to the nearest goal (inf = unreachable).

    cost[r, c] is the price of entering a cell; np.inf marks walls.
    goals: list of (row, col) cells, or a bool mask of goal cells.
    """
    n = cost.shape[0]
    if isinstance(goals, np.ndarray):
        src = np.flatnonzero(goals)
    else:
        src = [g[0] * n + g[1] for g in goals]
    if len(src) == 0:
        return np.full(cost.shape, np.inf)
    u, v, length, indptr = _graph(n)
    flat = cost.ravel()
    wall = ~np.isfinite(flat)
    w = 0.5 * (np.where(wall, 0.0, flat)[u] + np.where(wall, 0.0, flat)[v]) * length
    w[wall[u] | wall[v]] = WALL
    graph = csr_matrix((w, v, indptr), shape=(n * n, n * n))
    d = dijkstra(graph, directed=True, indices=src, min_only=True).reshape(n, n)
    d[d >= WALL] = np.inf
    return d


def step_down(field: np.ndarray, passable_mask: np.ndarray, pos: tuple[int, int]) -> tuple[int, int]:
    """Move one cell to the neighbour with the lowest field value (stay if none is lower)."""
    n = field.shape[0]
    best, best_v = pos, field[pos]
    for dr, dc in NEIGHBOURS:
        r, c = pos[0] + dr, pos[1] + dc
        if 0 <= r < n and 0 <= c < n and passable_mask[r, c] and field[r, c] < best_v:
            best, best_v = (r, c), field[r, c]
    return best
