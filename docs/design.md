# Design notes

## Coordinate conventions
World-to-camera `x_c = R X + t`; camera centre `C = -R^T t`; +z forward, +y down (OpenCV / COLMAP).

## Initial pair selection
For every matched pair (most matches first), estimate the essential matrix with RANSAC and
recover the pose. The score is `inliers * min(median_angle / min_init_angle, 1)`: many
inliers alone favour near-duplicate views with poorly conditioned depth, and a large angle
alone favours weakly matched views.

## Next-view selection
The unregistered image that observes the most already-reconstructed tracks is registered
next (COLMAP uses a refined version of this criterion that also considers the spatial
distribution of the observations). Images that fail PnP are retried after the next
successful registration, when more 3-D points may be available.

## Triangulation and filtering
A new track is triangulated from the current image and the registered partner with the
widest baseline. It is accepted only if the triangulation angle exceeds `min_tri_angle`,
it lies in front of every registered camera, and its reprojection error is below
`max_reproj_error` in every registered view. The same checks run after each bundle adjustment.

## Bundle adjustment
Parameters are 6 per free camera (rotation vector, translation) and 3 per point. The
Jacobian sparsity is declared explicitly, so SciPy's `trf` solver uses grouped finite
differences: the cost per iteration grows linearly with the number of observations. The
robust loss is applied in two stages (adaptive scale, then target scale).

With `refine_intrinsics="focal"` or `"radial"`, each distinct `Intrinsics` object adds a block of
1 or 3 parameters (`f`, `k1`, `k2`). The aspect ratio `fy / fx` and the principal point stay fixed:
without a calibration target the principal point is nearly unobservable and trades off against
rotation. Intrinsics are refined only once `min_images_for_intrinsics` views are registered,
since two or three views cannot separate focal length from depth. They converge more slowly
than poses, so the final BA gets a larger evaluation budget (`final_ba_max_nfev`). After every
self-calibrating BA the undistorted keypoints used by the minimal solvers are recomputed.

## Lens distortion
`Intrinsics.undistort` inverts `r_d = r_u (1 + k1 r_u² + k2 r_u⁴)` with Newton's method on the
radius, which converges in a few iterations for real lenses. The essential-matrix, PnP and DLT
solvers see undistorted pixels (`undistort_pixels`); reprojection checks and BA use the full
model against the raw detections, so no information is lost by resampling images.

## Retrieval
`BagOfVisualWords` learns a k-means vocabulary on RootSIFT descriptors (ORB descriptors are
unpacked to bits so Euclidean distance matches Hamming distance), weights word histograms by
IDF after a square-root (Hellinger) normalisation and ranks images by cosine similarity. The
neighbour relation from `retrieve_pairs` is symmetrised, so an image that is retrieved by
others but retrieves nobody is still matched.

## Renderer
`render.py` projects points with the same `Camera` class used for reconstruction, sorts
them far-to-near and splats discs with Pillow. The orbit axis is the mean camera "up"
direction, so turntables look right for reconstructions in any gauge.
