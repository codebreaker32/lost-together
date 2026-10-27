"""Common random numbers: one seed gives independent, named random streams.

The world, spawn and predator-start streams are consumed identically in every condition,
so a seed fixes the forest, exits, trailhead and predator start across solo/share/group.
Streams consumed during the episode (cues, noise, movement) diverge once behaviour differs;
that is unavoidable in a closed-loop simulation and is why results are paired by seed.
"""
import numpy as np

STREAMS = ("world", "spawn", "predator", "cues", "noise", "hikers")


def streams(seed: int) -> dict[str, np.random.Generator]:
    children = np.random.SeedSequence(seed).spawn(len(STREAMS))
    return {name: np.random.default_rng(child) for name, child in zip(STREAMS, children)}
