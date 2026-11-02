<div align="center">

# Structura

### Incremental Structure-from-Motion and Point-Cloud Processing, from First Principles

[Dang Truong Minh Le](https://github.com/dangaura-tech)

[![CI](https://github.com/dangaura-tech/structura/actions/workflows/ci.yml/badge.svg)](https://github.com/dangaura-tech/structura/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?logo=opencv&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)

<img src="assets/sfm_turntable.gif" width="640" alt="Sparse reconstruction turntable" />

<sub>Sparse reconstruction of a synthetic scene (sphere, box, torus, ground plane) from 12 views
with 10 % outlier matches. The orange wireframes are the recovered camera poses. The GIF is
rendered by the built-in software renderer: <code>python tools/demo_synthetic.py --gif assets/sfm_turntable.gif</code></sub>

### From pixels to cameras and points

<img src="assets/input_views.jpg" width="640" alt="Input views" />

<img src="assets/image_sfm_turntable.gif" width="560" alt="Reconstruction from images" />

<sub>Full image-based run: SIFT features → ratio-test matching → essential-matrix verification →
incremental SfM on 10 photo-like renders of a textured box on a textured floor.
Reproduce with <code>python tools/demo_images.py --assets assets</code>.</sub>

</div>

---

**Structura** turns a set of overlapping photos into camera poses and a coloured 3-D
point cloud. It implements the classic incremental SfM pipeline used by COLMAP and
OpenMVG in readable NumPy / SciPy / OpenCV, so every step can be inspected, modified and
unit-tested:

| Stage | Implementation |
|---|---|
| Features & matching | SIFT / ORB, Lowe ratio test, mutual nearest-neighbour check |
| Geometric verification | 5-point essential matrix + RANSAC per image pair |
| Feature tracks | union-find over verified matches; inconsistent tracks rejected |
| Initialisation | pair scored by inliers × triangulation-angle prior, cheirality-checked pose |
| Registration | PnP (EPnP + RANSAC) with iterative refinement, best-connected image first |
| Triangulation | multi-view DLT with widest-baseline partner, angle / depth / reprojection filters |
| Bundle adjustment | sparse Jacobian trust-region least squares, two-stage robust (Huber) loss |
| Point clouds | voxel downsampling, PCA normals, statistical outliers, point-to-point / point-to-plane ICP, RANSAC planes, Chamfer distance |
| I/O & visualisation | binary/ASCII PLY, COLMAP text model export, turntable GIF renderer |

## Pipeline

```mermaid
flowchart LR
    I[Images] --> F["SIFT / ORB<br/>features"]
    F --> M["Ratio test +<br/>mutual NN matching"]
    M --> V["Two-view verification<br/>E-matrix RANSAC"]
    V --> T["Feature tracks<br/>(union-find)"]
    T --> S["Initial pair<br/>pose + triangulation"]
    S --> R["Register next view<br/>PnP RANSAC"]
    R --> TR["Triangulate<br/>new tracks"]
    TR --> BA["Bundle adjustment<br/>+ outlier filtering"]
    BA -- more views --> R
    BA --> O["PLY · COLMAP model · GIF"]
```

## Installation

```bash
git clone https://github.com/dangaura-tech/structura.git
cd structura
pip install -e ".[dev]"
```

## Usage

**Reconstruct your own photos.** Walk around an object and take 15–50 overlapping photos.

```bash
python tools/reconstruct.py --images path/to/photos --out output/my_scene --gif
# -> output/my_scene/points.ply, output/my_scene/colmap/*.txt, output/my_scene/turntable.gif
```

The sparse model can be opened in [MeshLab](https://www.meshlab.net/) (`points.ply`) or
imported into the COLMAP GUI (`File → Import model`) for dense reconstruction.

**Image-based demo** (renders textured views, then runs the full pipeline from pixels):

```bash
python tools/demo_images.py --views 10 --out output/demo_images
```

**Synthetic benchmark with exact ground truth** (no data needed):

```bash
python tools/demo_synthetic.py --cameras 12 --noise 0.5 --outlier-ratio 0.1 --gif demo.gif
```

It reports registered views, reprojection error, camera position / rotation error and the
Chamfer distance to the true geometry, after a similarity alignment (Umeyama).

**Align two point clouds with ICP:**

```bash
python tools/register.py scan_a.ply scan_b.ply --out a_aligned.ply --voxel 0.02 --point-to-plane
```

**Python API:**

```python
from structura import IncrementalSfM, Intrinsics
from structura.features import detect, match_all

feats = {i: detect(img) for i, img in enumerate(images)}
intr = {i: Intrinsics.from_image_size(w, h) for i in feats}
rec = IncrementalSfM(intr, {i: f.keypoints for i, f in feats.items()}, match_all(feats)).run()
print(len(rec.cameras), "cameras,", rec.num_points, "points")
```

## Design notes

* **Why verify before building tracks?** A single outlier match can chain two unrelated
  keypoints into one track. The track then contains two observations in the same image and
  is discarded. Two-view RANSAC first keeps tracks long and consistent.
* **Gauge.** The first camera is fixed during bundle adjustment. The scale is set by the
  unit-norm baseline of the initial pair, so reconstructions are metric only up to a
  similarity transform.
* **Robust BA.** When the initialisation is poor, every residual is large and a robust loss
  with a small scale treats them all as outliers. Stage 1 therefore adapts the Huber scale
  to the median residual, and stage 2 switches to the target scale.

More details are in [`docs/design.md`](docs/design.md).

## Repository structure

```
structura/
├── camera.py          # pinhole model, poses, look-at, rotations
├── features.py        # SIFT/ORB detection and matching
├── geometry.py        # DLT triangulation, essential matrix, PnP, Umeyama
├── tracks.py          # union-find feature tracks
├── bundle.py          # sparse bundle adjustment
├── reconstruction.py  # incremental SfM driver
├── pointcloud.py      # voxels, normals, ICP, RANSAC planes, Chamfer
├── io.py              # PLY, COLMAP text export
├── render.py          # NumPy + Pillow point-cloud renderer, turntable GIFs
└── synthetic.py       # multi-view scenes with exact ground truth
tools/                 # reconstruct / demo_images / demo_synthetic / register
assets/                # README demo media (generated by the demo scripts)
tests/                 # geometry, SfM, point-cloud tests
```

## Roadmap

- [ ] Radial distortion in the camera model and BA
- [ ] Vocabulary-tree image retrieval for large unordered collections
- [ ] Multi-view stereo densification (plane sweep)
- [ ] Learned features (SuperPoint / LightGlue) as an optional front-end

## References

- Hartley & Zisserman, *Multiple View Geometry in Computer Vision*, 2004
- Schönberger & Frahm, *Structure-from-Motion Revisited* (COLMAP), CVPR 2016
- Nistér, *An Efficient Solution to the Five-Point Relative Pose Problem*, TPAMI 2004
- Lepetit et al., *EPnP*, IJCV 2009 · Triggs et al., *Bundle Adjustment - A Modern Synthesis*, 1999
- Besl & McKay, *ICP*, 1992 · Chen & Medioni, *point-to-plane ICP*, 1992 · Umeyama, TPAMI 1991

## License

MIT © Dang Truong Minh Le
