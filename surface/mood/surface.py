"""Fit one Gaussian surface to the paragraphs' (x, y, mood) points.

Gaussian Nadaraya-Watson (geometry.smoothers.gaussian_nw) with one bandwidth
for both axes, in standardized coordinates.
"""

import numpy as np

from geometry.smoothers import gaussian_nw


def _standardize(a):
    a = np.asarray(a, dtype=float)
    mean, std = a.mean(0), a.std(0)
    return mean, np.where(std > 1e-12, std, 1.0)


def fit(coords, values, h=0.2, resolution=120, margin=0.05, mask_floor=None):
    """Fit the surface and return (GX, GY, Z, height_at).

    `height_at(points)` evaluates the surface height at any 2D points.
    `mask_floor` is a support percentile below which cells are NaN; None or
    negative leaves the surface unbroken over the whole plane.
    """
    xm, xs = _standardize(coords)
    ym, ys = values.mean(), values.std() or 1.0
    Xn = (coords - xm) / xs
    yn = (values - ym) / ys

    lo, hi = Xn.min(0), Xn.max(0)
    pad = (hi - lo) * margin
    gx, gy = np.meshgrid(np.linspace(lo[0] - pad[0], hi[0] + pad[0], resolution),
                         np.linspace(lo[1] - pad[1], hi[1] + pad[1], resolution))
    Q = np.column_stack([gx.ravel(), gy.ravel()])

    vals, support = gaussian_nw(Xn, yn, Q, h, h)
    if mask_floor is not None and mask_floor >= 0:
        _, at_points = gaussian_nw(Xn, yn, Xn, h, h)
        vals = np.where(support >= np.percentile(at_points, mask_floor),
                        vals, np.nan)

    Z = (vals * ys + ym).reshape(gx.shape)
    GX = (gx * xs[0] + xm[0])
    GY = (gy * xs[1] + xm[1])

    def height_at(points):
        pn = (np.asarray(points, dtype=float) - xm) / xs
        v, _ = gaussian_nw(Xn, yn, pn, h, h)
        return np.clip(v * ys + ym, 0.0, 1.0)

    return GX, GY, Z, height_at
