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

## Renderer
`render.py` projects points with the same `Camera` class used for reconstruction, sorts
them far-to-near and splats discs with Pillow. The orbit axis is the mean camera "up"
direction, so turntables look right for reconstructions in any gauge.
