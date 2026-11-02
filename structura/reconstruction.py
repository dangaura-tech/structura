"""Incremental Structure-from-Motion.

1. Build feature tracks from pairwise matches.
2. Initialise from the image pair with the most essential-matrix inliers whose
   triangulation angles are large enough (a wide but well-matched baseline).
3. Repeatedly register the unregistered image that sees the most reconstructed points
   (PnP + RANSAC), triangulate new tracks it shares with registered images, and filter
   points with large reprojection error or small triangulation angle.
4. Run global bundle adjustment after every few registrations and at the end.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .bundle import BAResult, bundle_adjust
from .camera import Camera, Intrinsics
from .geometry import absolute_pose, relative_pose, reprojection_errors, triangulate_dlt, triangulation_angles
from .tracks import build_tracks

log = logging.getLogger("structura")


@dataclass
class SfMConfig:
    min_init_inliers: int = 50
    min_init_angle: float = 2.0  # median triangulation angle (deg) of the initial pair
    min_tri_angle: float = 1.5
    max_reproj_error: float = 4.0
    ransac_threshold: float = 1.0
    pnp_threshold: float = 4.0
    min_pnp_inliers: int = 15
    ba_every: int = 3
    ba_max_nfev: int = 50
    verify_matches: bool = True
    min_verified_matches: int = 30


@dataclass
class Reconstruction:
    cameras: Dict[int, Camera] = field(default_factory=dict)
    points: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    colors: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.uint8))
    point_tracks: List[Dict[int, int]] = field(default_factory=list)  # per point: image -> keypoint

    @property
    def num_points(self) -> int:
        return len(self.points)

    def observations(self, keypoints: Dict[int, np.ndarray]):
        cam_idx, pt_idx, obs = [], [], []
        for p, track in enumerate(self.point_tracks):
            for img, kp in track.items():
                if img in self.cameras:
                    cam_idx.append(img)
                    pt_idx.append(p)
                    obs.append(keypoints[img][kp])
        return np.array(cam_idx, int), np.array(pt_idx, int), np.array(obs, float).reshape(-1, 2)

    def mean_reprojection_error(self, keypoints: Dict[int, np.ndarray]) -> float:
        cam_idx, pt_idx, obs = self.observations(keypoints)
        if not len(obs):
            return float("nan")
        errs = [
            reprojection_errors(self.cameras[c], self.points[p][None], o[None])[0]
            for c, p, o in zip(cam_idx, pt_idx, obs, strict=True)
        ]
        return float(np.mean(errs))


class IncrementalSfM:
    def __init__(
        self,
        intrinsics: Dict[int, Intrinsics],
        keypoints: Dict[int, np.ndarray],
        matches: Dict[Tuple[int, int], np.ndarray],
        config: Optional[SfMConfig] = None,
        colors: Optional[Dict[int, np.ndarray]] = None,
    ) -> None:
        self.K = intrinsics
        self.kps = {i: np.asarray(k, float) for i, k in keypoints.items()}
        self.cfg = config or SfMConfig()
        raw = {k: np.asarray(v, int) for k, v in matches.items() if len(v)}
        self.matches = self._verify(raw) if self.cfg.verify_matches else raw
        self.kp_colors = colors or {}
        self.tracks = build_tracks(self.matches)
        self.rec = Reconstruction()
        self._track_point: Dict[int, int] = {}  # track index -> point index
        self.ba_history: List[BAResult] = []
        self._failed: set = set()
        log.info("%d images, %d pairs, %d tracks", len(self.kps), len(self.matches), len(self.tracks))

    def _verify(self, raw: Dict[Tuple[int, int], np.ndarray]) -> Dict[Tuple[int, int], np.ndarray]:
        """Two-view geometric verification: keep only essential-matrix RANSAC inliers.

        Unverified outlier matches would chain unrelated keypoints into the same track,
        which then gets discarded as inconsistent, so this step is essential for track quality.
        """
        out = {}
        for (i, j), m in raw.items():
            if len(m) < self.cfg.min_verified_matches:
                continue
            res = relative_pose(self.kps[i][m[:, 0]], self.kps[j][m[:, 1]], self.K[i], self.cfg.ransac_threshold)
            if res is not None and res.num_inliers >= self.cfg.min_verified_matches:
                out[(i, j)] = m[res.inliers]
        log.info("geometric verification kept %d / %d pairs", len(out), len(raw))
        return out

    # ------------------------------------------------------------------ helpers
    def _track_obs(self, t: int, img: int) -> Optional[np.ndarray]:
        kp = self.tracks[t].get(img)
        return None if kp is None else self.kps[img][kp]

    def _color(self, track: Dict[int, int]) -> np.ndarray:
        cols = [self.kp_colors[i][k] for i, k in track.items() if i in self.kp_colors]
        return np.mean(cols, axis=0).astype(np.uint8) if cols else np.array([200, 200, 200], np.uint8)

    def _add_points(self, track_ids: List[int], X: np.ndarray) -> None:
        start = len(self.rec.points)
        self.rec.points = np.vstack([self.rec.points, X])
        self.rec.colors = np.vstack([self.rec.colors, [self._color(self.tracks[t]) for t in track_ids]])
        for k, t in enumerate(track_ids):
            self._track_point[t] = start + k
            self.rec.point_tracks.append(self.tracks[t])

    # ------------------------------------------------------------- initialisation
    def _initialize(self) -> Tuple[int, int]:
        best = None
        for (i, j), m in sorted(self.matches.items(), key=lambda kv: -len(kv[1])):
            if len(m) < self.cfg.min_init_inliers:
                continue
            x1, x2 = self.kps[i][m[:, 0]], self.kps[j][m[:, 1]]
            res = relative_pose(x1, x2, self.K[i], self.cfg.ransac_threshold)  # assumes shared K for the pair
            if res is None or res.num_inliers < self.cfg.min_init_inliers:
                continue
            angle = np.median(triangulation_angles(np.zeros(3), -res.R.T @ res.t, res.points))
            score = res.num_inliers * min(angle / self.cfg.min_init_angle, 1.0)
            if best is None or score > best[0]:
                best = (score, i, j, res)
        if best is None:
            raise RuntimeError("no image pair is suitable for initialisation")
        _, i, j, res = best
        self.rec.cameras[i] = Camera(self.K[i])
        self.rec.cameras[j] = Camera(self.K[j], res.R, res.t)
        log.info("initial pair (%d, %d) with %d inliers", i, j, res.num_inliers)
        self._triangulate_new(j)
        return i, j

    # --------------------------------------------------------------- registration
    def _next_image(self) -> Optional[Tuple[int, List[int]]]:
        best: Optional[Tuple[int, List[int]]] = None
        for img in self.kps:
            if img in self.rec.cameras or img in self._failed:
                continue
            seen = [t for t in self._track_point if img in self.tracks[t]]
            if len(seen) >= self.cfg.min_pnp_inliers and (best is None or len(seen) > len(best[1])):
                best = (img, seen)
        return best

    def _register(self, img: int, track_ids: List[int]) -> bool:
        X = self.rec.points[[self._track_point[t] for t in track_ids]]
        x = np.stack([self._track_obs(t, img) for t in track_ids])
        res = absolute_pose(X, x, self.K[img], self.cfg.pnp_threshold)
        if res is None or res[2].sum() < self.cfg.min_pnp_inliers:
            return False
        self.rec.cameras[img] = Camera(self.K[img], res[0], res[1])
        log.info("registered image %d (%d / %d PnP inliers)", img, int(res[2].sum()), len(track_ids))
        return True

    def _triangulate_new(self, img: int) -> int:
        """Triangulate tracks seen by ``img`` and at least one other registered image."""
        new_ids, new_X = [], []
        cam = self.rec.cameras[img]
        for t, track in enumerate(self.tracks):
            if t in self._track_point or img not in track:
                continue
            others = [o for o in track if o != img and o in self.rec.cameras]
            if not others:
                continue
            # use the partner with the widest baseline
            other = max(others, key=lambda o: np.linalg.norm(self.rec.cameras[o].center - cam.center))
            c2 = self.rec.cameras[other]
            X = triangulate_dlt(cam.P, c2.P, self._track_obs(t, img)[None], self._track_obs(t, other)[None])
            if not self._valid(X, track, (cam.center, c2.center)):
                continue
            new_ids.append(t)
            new_X.append(X[0])
        if new_ids:
            self._add_points(new_ids, np.array(new_X))
        return len(new_ids)

    def _valid(self, X: np.ndarray, track: Dict[int, int], centers) -> bool:
        if triangulation_angles(centers[0], centers[1], X)[0] < self.cfg.min_tri_angle:
            return False
        for o, kp in track.items():
            c = self.rec.cameras.get(o)
            if c is None:
                continue
            if c.depth(X)[0] <= 0 or reprojection_errors(c, X, self.kps[o][kp][None])[0] > self.cfg.max_reproj_error:
                return False
        return True

    # ---------------------------------------------------------------- refinement
    def bundle_adjust(self, fixed_first: int) -> BAResult:
        cam_idx, pt_idx, obs = self.rec.observations(self.kps)
        res = bundle_adjust(
            self.rec.cameras, self.rec.points, cam_idx, pt_idx, obs, fixed=[fixed_first], max_nfev=self.cfg.ba_max_nfev
        )
        self.ba_history.append(res)
        self._filter_points()
        log.info("bundle adjustment: rmse %.3f -> %.3f px", res.initial_rmse, res.final_rmse)
        return res

    def _filter_points(self) -> None:
        keep = []
        for p, track in enumerate(self.rec.point_tracks):
            X = self.rec.points[p][None]
            ok = all(
                self.rec.cameras[o].depth(X)[0] > 0
                and reprojection_errors(self.rec.cameras[o], X, self.kps[o][kp][None])[0] <= self.cfg.max_reproj_error
                for o, kp in track.items()
                if o in self.rec.cameras
            )
            keep.append(ok)
        keep = np.array(keep, bool)
        if keep.all():
            return
        old_to_new = np.cumsum(keep) - 1
        self.rec.points = self.rec.points[keep]
        self.rec.colors = self.rec.colors[keep]
        self.rec.point_tracks = [t for t, k in zip(self.rec.point_tracks, keep, strict=True) if k]
        self._track_point = {t: int(old_to_new[p]) for t, p in self._track_point.items() if keep[p]}

    # ------------------------------------------------------------------- driver
    def run(self) -> Reconstruction:
        first, _ = self._initialize()
        self.bundle_adjust(first)
        since_ba = 0
        while True:
            nxt = self._next_image()
            if nxt is None:
                break
            img, seen = nxt
            if not self._register(img, seen):
                self._failed.add(img)  # retried after the next successful registration
                continue
            self._failed.clear()
            self._triangulate_new(img)
            since_ba += 1
            if since_ba >= self.cfg.ba_every:
                self.bundle_adjust(first)
                since_ba = 0
        self.bundle_adjust(first)
        log.info(
            "done: %d / %d images registered, %d points", len(self.rec.cameras), len(self.kps), self.rec.num_points
        )
        return self.rec
