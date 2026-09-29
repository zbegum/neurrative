"""A tube around the narrative arc: a polygon cross-section at every window,
swept smoothly along the fitted curve.

The arc says where the story goes. The tube adds, at every moment, a shape
describing what the story is like there -- so the arc becomes a surface whose
thickness changes in time.

Construction:

  1. Curve. The B-spline fitted through the windows (curves.fit_curve), sampled
     densely in reading order.

  2. Frames. A rotation-minimizing frame (T, N, B) at every sample, by the double
     reflection method (Wang, Juttler, Zheng, Liu, "Computation of rotation
     minimizing frames", ACM TOG 2008). Frenet frames flip wherever the curvature
     changes sign, which would spin every polygon half a turn; a rotation-
     minimizing frame turns only as much as the curve forces it to, so "vertex j"
     points the same way from one section to the next.

  3. Sections. k radii per window, one per vertex, vertex j at angle 2 pi j / k in
     the (N, B) plane. Two sources:

       emotions  vertex j = emotion j; its radius grows with the window's pooled
                 score, so the tube bulges on the side of whichever emotion is
                 strong. Radii are in fractions of the layout's RMS radius.

       spread    from the embeddings (PCA only): the window's paragraphs, placed
                 with the same PCA, projected onto the window's (N, B) plane; the
                 section is their covariance ellipse, sampled at k angles. By
                 default it is scaled as a standard error (sd / sqrt(paragraphs)),
                 i.e. how precisely the window's point is located -- in real PCA
                 units, no arbitrary scale.

  4. Sweep. Each window sits at its fitted position on the curve. Every vertex's
     radius is smoothed across windows (Gaussian, in windows) and interpolated to
     every curve sample with PCHIP, which is smooth and never overshoots -- so a
     radius never goes negative or bulges past its neighbours.

  5. No folding. Where the curve bends tighter than the tube is thick (radius
     above 1 / curvature), neighbouring sections cross and the surface folds
     through itself. Thickness is the signal, so it must stay comparable along
     the whole tube: first the *entire* tube is scaled by one factor, chosen so
     that 95% of the curve is fold-free; only the sharpest remaining bends are
     then narrowed locally (each section uniformly, so its shape is kept). Both
     the global factor and the locally narrowed fraction are reported.

  6. Mesh. Rings of k (or `round`) vertices joined by quads split into
     triangles, capped at both ends.
"""

from collections import namedtuple

import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.ndimage import gaussian_filter1d

# vertices (V, 3), faces (F, 3) int. rings: (S, m, 3), the swept sections without
# caps. radii: (S, m) at every curve sample. window_index: (n,) curve sample of
# each window. window_rings: (n, k, 3) the unsmoothed polygon at each window.
# angles: (m,). labels: one name per original vertex, or None.
# scale: the one factor applied to every section to avoid folding (1 = none).
# clamped: fraction of curve samples additionally narrowed at a sharp bend.
Tube = namedtuple("Tube", "vertices faces rings radii window_index window_rings "
                          "angles labels scale clamped")


# --------------------------------------------------------------------------
# frames
# --------------------------------------------------------------------------

def tangents(points):
  t = np.gradient(points, axis=0)
  return t / np.clip(np.linalg.norm(t, axis=1, keepdims=True), 1e-12, None)


def curvature(points):
  """Discrete curvature |dT/ds| at every sample."""
  T = tangents(points)
  ds = np.linalg.norm(np.gradient(points, axis=0), axis=1)
  return np.linalg.norm(np.gradient(T, axis=0), axis=1) / np.clip(ds, 1e-12, None)


def limit_folding(radii, points, fold_limit=0.9, coverage=0.95):
  """Keep max radius <= fold_limit / curvature. Returns (radii, scale, clamped).

  First one global factor, the largest that makes `coverage` of the samples
  fold-free, so relative thickness along the tube is untouched. Then any sample
  still over the limit is narrowed locally; that local factor is smoothed along
  the curve (running minimum, light blur) so a tight bend narrows the tube
  gradually instead of pinching it at one sample.
  """
  kappa = curvature(points)
  allowed = fold_limit / np.clip(kappa, 1e-12, None)
  ratio = allowed / np.clip(radii.max(axis=1), 1e-12, None)
  scale = float(min(1.0, np.quantile(ratio, 1.0 - coverage)))
  radii = radii * scale

  local = np.minimum(1.0, ratio / scale)
  width = max(3, len(points) // 100)
  padded = np.pad(local, width, mode="edge")
  windows = np.lib.stride_tricks.sliding_window_view(padded, 2 * width + 1)
  local = np.minimum(local, gaussian_filter1d(windows.min(axis=1), width / 2))
  return radii * local[:, None], scale, float((local < 0.999).mean())


def rotation_minimizing_frames(points):
  """(T, N, B), each (S, 3), by double reflection (Wang et al. 2008)."""
  T = tangents(points)
  S = len(points)
  N = np.empty_like(points)
  B = np.empty_like(points)

  # Any normal will do at the start: take the axis least aligned with T.
  seed = np.eye(3)[np.argmin(np.abs(T[0]))]
  N[0] = seed - seed.dot(T[0]) * T[0]
  N[0] /= np.linalg.norm(N[0])
  B[0] = np.cross(T[0], N[0])

  for i in range(S - 1):
    v1 = points[i + 1] - points[i]
    c1 = v1.dot(v1)
    if c1 < 1e-20:
      N[i + 1], B[i + 1] = N[i], B[i]
      continue
    rL = N[i] - (2.0 / c1) * v1.dot(N[i]) * v1
    tL = T[i] - (2.0 / c1) * v1.dot(T[i]) * v1
    v2 = T[i + 1] - tL
    c2 = v2.dot(v2)
    r = rL if c2 < 1e-20 else rL - (2.0 / c2) * v2.dot(rL) * v2
    r -= r.dot(T[i + 1]) * T[i + 1]  # scrub round-off drift
    N[i + 1] = r / np.linalg.norm(r)
    B[i + 1] = np.cross(T[i + 1], N[i + 1])
  return T, N, B


# --------------------------------------------------------------------------
# where each window sits on the curve
# --------------------------------------------------------------------------

def window_positions(fit, n_samples, degree=3):
  """(fraction along the curve, curve-sample index) of each window, from its
  fitted correspondence u.

  An open uniform B-spline with m control points is parameterised over
  [0, m - degree), and the curve samples span that whole range, so u / (m -
  degree) is the fraction along the samples. The solver can leave a window's u
  slightly behind its predecessor's; the tube is swept in reading order, so
  positions are made non-decreasing first.
  """
  span = len(fit.control_points) - degree
  frac = np.clip(np.maximum.accumulate(fit.u) / span, 0.0, 1.0)
  return frac, np.round(frac * (n_samples - 1)).astype(int)


# --------------------------------------------------------------------------
# sections
# --------------------------------------------------------------------------

def polygon_angles(k):
  return 2.0 * np.pi * np.arange(k) / k


def emotion_sections(series, layout_radius, thickness=0.12, core=0.15,
                     normalize=False):
  """(n, k) radii from pooled emotion scores, in data units.

  radius = thickness * layout_radius * (core + (1 - core) * score), so a zero
  score still leaves a thin core instead of pinching the tube to a line.
  `normalize` rescales each emotion to its own [0, 1] range over the book, to
  show how each one changes rather than how strong it is.
  """
  if series.scores is None:
    raise ValueError(f"{series.book} has no emotion scores; use --section spread.")
  scores = np.asarray(series.scores, dtype=float)
  if normalize:
    lo, hi = scores.min(axis=0), scores.max(axis=0)
    scores = (scores - lo) / np.where(hi > lo, hi - lo, 1.0)
  radii = thickness * layout_radius * (core + (1.0 - core) * scores)
  return radii, list(series.emotions)


def spread_sections(projection, embeddings, series, N, B, window_index, k=12,
                    scale="se", thickness=1.0):
  """(n, k) radii: each window's paragraph scatter, perpendicular to the curve.

  The paragraphs of window w are placed with the same PCA as the windows, their
  offsets from the window mean are projected onto (N, B) at the window, and the
  section is the covariance ellipse r(theta) = 1 / sqrt(u^T C^-1 u) sampled at k
  angles. scale "sd" is one standard deviation of the paragraphs; "se" divides by
  sqrt(paragraphs in the window) -- the uncertainty of the window's own point.
  """
  if projection.model is None:
    raise ValueError(
      "spread sections need PCA: single paragraphs must be placed in the same "
      "space as the windows, and only PCA is a map that can place new points.")
  placed = projection.model.transform(embeddings)
  theta = polygon_angles(k)
  dirs = np.stack([np.cos(theta), np.sin(theta)], axis=1)  # (k, 2)
  radii = np.empty((len(series), k))

  for w, (start, stop) in enumerate(series.windows):
    offsets = placed[start:stop] - placed[start:stop].mean(axis=0)
    i = window_index[w]
    plane = offsets @ np.stack([N[i], B[i]], axis=1)  # (m, 2)
    C = np.cov(plane, rowvar=False) + 1e-12 * np.eye(2)
    inv = np.linalg.inv(C)
    r = 1.0 / np.sqrt(np.einsum("kj,jl,kl->k", dirs, inv, dirs))
    if scale == "se":
      r = r / np.sqrt(stop - start)
    radii[w] = thickness * r
  return radii, None


# --------------------------------------------------------------------------
# sweep and mesh
# --------------------------------------------------------------------------

def sweep_radii(window_frac, radii, n_samples, smooth=1.0):
  """(S, k): every vertex's radius at every curve sample.

  Smoothed over windows with a Gaussian of `smooth` windows (0 = off), then
  interpolated with PCHIP. Windows that share a position (after the positions
  were made non-decreasing) are averaged first, since PCHIP needs distinct knots.
  """
  if smooth > 0:
    radii = gaussian_filter1d(radii, smooth, axis=0, mode="nearest")
  knots, inverse = np.unique(window_frac, return_inverse=True)
  values = np.zeros((len(knots), radii.shape[1]))
  np.add.at(values, inverse, radii)
  values /= np.bincount(inverse)[:, None]

  tau = np.linspace(0.0, 1.0, n_samples)
  if len(knots) == 1:
    return np.repeat(values, n_samples, axis=0)
  inside = np.clip(tau, knots[0], knots[-1])  # hold the end sections flat
  return np.clip(PchipInterpolator(knots, values, axis=0)(inside), 0.0, None)


def round_sections(radii, m):
  """Upsample k radii per section to m with a periodic cubic spline: a rounded
  tube through the same vertex radii. m = 0 keeps the polygon."""
  k = radii.shape[1]
  if not m or m <= k:
    return radii, polygon_angles(k)
  theta = polygon_angles(k)
  closed = np.concatenate([radii, radii[:, :1]], axis=1)
  spline = CubicSpline(np.append(theta, 2 * np.pi), closed, axis=1,
                       bc_type="periodic")
  fine = polygon_angles(m)
  return np.clip(spline(fine), 0.0, None), fine


def rings(curve, N, B, radii, angles):
  """(S, m, 3): the section at every sample, placed in its (N, B) plane."""
  c, s = np.cos(angles), np.sin(angles)
  offset = (radii[..., None]
            * (c[None, :, None] * N[:, None, :] + s[None, :, None] * B[:, None, :]))
  return curve[:, None, :] + offset


def mesh(ring_points, curve, caps=True):
  """Triangle mesh from (S, m, 3) rings: quads between rings, fans at the ends."""
  S, m, _ = ring_points.shape
  vertices = ring_points.reshape(-1, 3)
  idx = np.arange(S * m).reshape(S, m)
  a, b = idx[:-1], idx[1:]
  a2, b2 = np.roll(a, -1, axis=1), np.roll(b, -1, axis=1)
  faces = np.concatenate([
    np.stack([a, b, b2], axis=-1).reshape(-1, 3),
    np.stack([a, b2, a2], axis=-1).reshape(-1, 3),
  ])
  if caps:
    start, end = len(vertices), len(vertices) + 1
    vertices = np.vstack([vertices, curve[0], curve[-1]])
    ring0, ring1 = idx[0], idx[-1]
    faces = np.concatenate([
      faces,
      np.stack([np.full(m, start), np.roll(ring0, -1), ring0], axis=1),
      np.stack([np.full(m, end), ring1, np.roll(ring1, -1)], axis=1),
    ])
  return vertices, faces


def build_tube(fit, window_radii, degree=3, labels=None, smooth=1.0, round_to=0,
               frames=None, fold_limit=0.9):
  """Sweep window sections along a Fit. Returns a Tube."""
  curve = fit.curve
  S = len(curve)
  _, N, B = frames if frames is not None else rotation_minimizing_frames(curve)
  frac, index = window_positions(fit, S, degree)

  k = window_radii.shape[1]
  window_ring = rings(curve[index], N[index], B[index], window_radii,
                      polygon_angles(k))

  swept = sweep_radii(frac, window_radii, S, smooth)
  swept, angles = round_sections(swept, round_to)
  scale, clamped = 1.0, 0.0
  if fold_limit:
    swept, scale, clamped = limit_folding(swept, curve, fold_limit)
    window_ring = (curve[index][:, None, :]
                   + scale * (window_ring - curve[index][:, None, :]))
  ring_points = rings(curve, N, B, swept, angles)
  vertices, faces = mesh(ring_points, curve)
  return Tube(vertices, faces, ring_points, swept, index, window_ring, angles,
              labels, scale, clamped)
