"""Feature tracks: connected components of pairwise matches across images (union-find)."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

Observation = Tuple[int, int]  # (image id, keypoint id)


class UnionFind:
    def __init__(self) -> None:
        self.parent: Dict[Observation, Observation] = {}

    def find(self, a: Observation) -> Observation:
        self.parent.setdefault(a, a)
        root = a
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[a] != root:  # path compression
            self.parent[a], a = root, self.parent[a]
        return root

    def union(self, a: Observation, b: Observation) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def build_tracks(matches: Dict[Tuple[int, int], np.ndarray], min_length: int = 2) -> List[Dict[int, int]]:
    """Merge pairwise matches ``{(i, j): array[M, 2] of (kp_i, kp_j)}`` into tracks.

    A track maps image id -> keypoint id. Tracks that contain two different keypoints of the
    same image are inconsistent (a match chain loops back) and are discarded.
    """
    uf = UnionFind()
    for (i, j), m in matches.items():
        for a, b in np.asarray(m, dtype=int):
            uf.union((i, int(a)), (j, int(b)))
    groups: Dict[Observation, List[Observation]] = {}
    for obs in list(uf.parent):
        groups.setdefault(uf.find(obs), []).append(obs)
    tracks = []
    for members in groups.values():
        images = [img for img, _ in members]
        if len(set(images)) != len(images) or len(members) < min_length:
            continue
        tracks.append({img: kp for img, kp in members})
    return tracks
