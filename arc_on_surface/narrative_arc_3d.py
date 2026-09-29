"""
The narrative arc lifted into 3-D: emotion is the height.

The 2-D arc (arc/curve/arc_2d.py) draws the sliding-window path on a flat
projection, where x and y are arbitrary UMAP/t-SNE (or PCA) coordinates with no
units. Here the third axis carries a real, metric quantity so the story becomes
a curve rising and falling over the semantic map. What that quantity is is up
to --z-mode:

  dominant (default)  the winning emotion's score (0..1) -- height is intensity,
                      and each point's color tells you which emotion won.
  spectrum            the emotions laid out in a fixed mood order (dark->light);
                      the height is the score-weighted position of the window's
                      emotional blend on that line, and the z-ticks are the
                      emotion names -- so you read the emotion straight off z.
  an emotion name     that one emotion's score (0..1) -- a single clean arc,
                      e.g. --z-mode sadness for the sadness arc.
  pc1                 the first principal component of the six emotions: the one
                      axis explaining the most emotional variation in this book,
                      normalized to [-1, 1] and oriented so its strongest emotion
                      is the positive pole. This is the "read the emotion off the
                      height" axis -- the printed loadings say what the poles are.

Six emotions are six numbers per window and z is one, so every mode is a lossy
collapse; the marker color always carries the dominant emotion regardless.

This is the score-z idea from emotion_3d.py / surface/kernel/loo.py, applied to the
windowed arc instead of to raw paragraphs. As there, only PCA is a genuine
height field (two windows can share a UMAP/t-SNE (x, y) at different heights), so
for the PCA base we also fit and draw the emotion terrain with the same kernel
smoother the geodesic work uses, and let the arc ride it. UMAP and t-SNE get the
bare 3-D arc, no surface.

The height itself is a *smooth* emotional signal over reading position, not a
window-by-window step. Rather than averaging each window's paragraphs in a box
(a rectangular kernel with hard edges at the window bounds), every emotion is
regressed against paragraph index with a Gaussian kernel -- the same
Nadaraya-Watson smoother scalar_field.py uses over the plane, here in 1-D -- and
read off at each window's center. So the arc's rise and fall no longer jump when
a window edge crosses a sharp paragraph. Pass --z-bandwidth pool to fall back to
the old boxcar height for comparison.

Read the caveat honestly: x and y are non-metric, z is metric, so only height is
signal -- 3-D distances and slopes between points mean nothing. It is a path
over a landscape, not a shape.

Each projection writes a static PNG and, if plotly is installed, a rotatable
HTML. Everything lands in arc_on_surface/output/<book>/<model>/narrative_arc_3d/.

Examples:

python arc_on_surface/narrative_arc_3d.py --book alice_wonderland --model bge-m3
python arc_on_surface/narrative_arc_3d.py --book alice_wonderland --model bge-m3 \
  --z-mode pc1 --proj pca
python arc_on_surface/narrative_arc_3d.py --book alice_wonderland --model bge-m3 \
  --size 25 --stride 12 --proj pca umap tsne
"""

import os
import sys
import argparse

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP


# The repository root and common/ on the path, wherever this script lives.
_ROOT = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_ROOT, "common")):
  _ROOT = os.path.dirname(_ROOT)
sys.path[1:1] = [_ROOT, os.path.join(_ROOT, "common")]

import paths
from data import UNCLEAR, dominant, load_paragraphs, load_scores
from windows import pool as pool_embeddings, window_bounds, window_centers
from geometry.scalar_field import (
  bandwidth_candidates, loo_bandwidth, nadaraya_watson,
)
from geometry.smoothers import SMOOTHERS, gaussian_nw
from geometry.geodesic import exact_paths
from geometry.mesh import edge_graph, height_mesh, snap
from smooth_common import Standardizer, fit_grid, tune
from surface.mood.mood import SPECTRUM
from surface.kernel.gaussian import GRID as GAUSSIAN_GRID
from surface.kernel.epanechnikov import GRID as EPANECHNIKOV_GRID
from surface.kernel.local_linear import GRID as LOCAL_LINEAR_GRID
from surface.kernel.loess import GRID as LOESS_GRID

try:
  import plotly.graph_objects as go
  from plotly.subplots import make_subplots
  HAS_PLOTLY = True
except ImportError:
  HAS_PLOTLY = False

CMAP = "viridis"

# The six emotions laid out as a mood spectrum, darkest (bottom) to lightest
# (top). The z = "spectrum" mode reads each window's height as the score-weighted
# position of its emotional blend along this line, so low = dark/tense stretches,
# high = light/funny ones, and a mixed stretch sits in between. The z-ticks are
# labelled with these names, so the height is read directly off the spectrum.
# The same order as every other mood axis in the repository (surface/mood/mood.py).
EMOTION_SPECTRUM = SPECTRUM

# The terrain smoothers, with the same tuning ladders bandwidth_grid.py uses --
# hx/hy are in standard deviations of the standardized coordinates, so a ladder
# means the same thing on any projection or book. loess adds its robustness
# rounds. local_linear is the default: it fits a tilted plane at each query, so
# it follows the trend off the edge of the data instead of flattening there --
# the boundary bias that matters most where the arc's opening and ending sit.
TERRAIN_GRIDS = {
  "gaussian_nw": GAUSSIAN_GRID,
  "epanechnikov_nw": EPANECHNIKOV_GRID,
  "local_linear": LOCAL_LINEAR_GRID,
  "loess": dict(LOESS_GRID, iterations=[2]),
}


def project(embeddings, proj, args):
  """2-D layout of the pooled windows. Returns (coords, xlabel, ylabel, tag)."""
  if proj == "pca":
    pca = PCA(n_components=2)
    coords = pca.fit_transform(embeddings)
    var = pca.explained_variance_ratio_
    print(f"  PCA: {var.sum():.1%} var over 2 components")
    return coords, f"PC1 ({var[0]:.1%} var)", f"PC2 ({var[1]:.1%} var)", ""

  if proj == "umap":
    coords = UMAP(n_components=2, n_neighbors=args.neighbors,
                  min_dist=args.min_dist, random_state=args.seed).fit_transform(embeddings)
    return coords, "UMAP-1", "UMAP-2", f"_n{args.neighbors}_d{args.min_dist}_s{args.seed}"

  perplexity = min(args.perplexity, len(embeddings) - 1)
  coords = TSNE(n_components=2, perplexity=perplexity,
                random_state=args.seed).fit_transform(embeddings)
  return coords, "t-SNE-1", "t-SNE-2", f"_p{perplexity}_s{args.seed}"


def smooth_emotions(matrix, centers, bandwidth):
  """Each emotion as a continuous curve over reading position, read off at the
  window centers.

  Instead of averaging a window's paragraphs in a box (a rectangular kernel with
  hard edges at the window bounds), regress every emotion against paragraph index
  with a Gaussian kernel -- Nadaraya-Watson in 1-D, the same smoother
  scalar_field.py runs over the plane -- and sample it at each window's center.
  The result is a smooth emotional signal: its rise and fall no longer jump when
  a window edge steps across a sharp paragraph.

  `bandwidth` is in paragraphs: the Gaussian's sigma over reading position.
  """
  pos = np.arange(len(matrix), dtype=float).reshape(-1, 1)
  q = np.asarray(centers, dtype=float).reshape(-1, 1)
  cols = [nadaraya_watson(pos, matrix[:, c], q, bandwidth)[0]
          for c in range(matrix.shape[1])]
  return np.column_stack(cols)


def loo_emotion_bandwidth(matrix):
  """One bandwidth (in paragraphs) for every emotion, by leave-one-out CV.

  A shared scale keeps the six heights comparable; it is the bandwidth on a
  geometric ladder that minimizes the summed LOO error across the emotions. Note
  this tunes for *fidelity to the noisy per-paragraph scores*, so it can land
  well below the window width -- a signal that fits the annotations tightly, not
  necessarily one smoother than the boxcar. Use it when you want the data-driven
  scale rather than the window's; "auto" (window/2) is the smoother default.
  """
  pos = np.arange(len(matrix), dtype=float).reshape(-1, 1)
  candidates = bandwidth_candidates(pos)
  total = np.zeros(len(candidates))
  for c in range(matrix.shape[1]):
    _, mses = loo_bandwidth(pos, matrix[:, c], candidates)
    total += mses
  return float(candidates[int(np.argmin(total))])


def window_emotions(book, paragraphs, windows, min_score, bandwidth):
  """Smoothed per-window emotion matrix, plus the dominant-emotion color codes.

  `bandwidth` chooses how each emotion is read at each window:
    None    -- block-mean pool over the window (the original step-like boxcar);
    "auto"  -- Gaussian over reading position at sigma = window/2, a smooth
               analog of the box at the same scale (the sensible default);
    "loo"   -- Gaussian at the leave-one-out-CV bandwidth (data-driven, can be
               narrower than the window and so *rougher* -- see the caveat above);
    a float -- Gaussian at that bandwidth in paragraphs.

  The color channel (which emotion wins) is always the dominant emotion,
  whatever the z-axis is set to carry -- only the height changes with --z-mode,
  in z_axis. Returns (emotions, pooled, codes, categories).
  """
  emotions, matrix = load_scores(book, paragraphs)

  if bandwidth is None:
    pooled = np.array([matrix[s:e].mean(axis=0) for s, e in windows])
  else:
    if bandwidth == "auto":
      size = windows[0][1] - windows[0][0]
      bandwidth = size / 2.0
    elif bandwidth == "loo":
      bandwidth = loo_emotion_bandwidth(matrix)
    pooled = smooth_emotions(matrix, window_centers(windows), bandwidth)
    print(f"  emotion signal: Gaussian over reading position, "
          f"sigma {bandwidth:.1f} paragraphs")

  codes, categories, n_unclear = dominant(emotions, pooled, min_score)
  print(f"  dominant: {n_unclear} of {len(codes)} windows unclear")
  return emotions, pooled, codes, categories


def z_axis(emotions, pooled, z_mode):
  """The height each window sits at, and how to label and scale that axis.

  The six emotions are six numbers per window; the z-axis is one. This chooses
  how to collapse them -- and so what reading the height actually tells you.
  Marker *color* is separate (always the dominant emotion), so only height's
  meaning changes here. Returns (height, label, zlim, clim, zticks): zlim the
  axis range, clim the terrain colormap range, and zticks either None or a
  (positions, labels) pair to label the axis with (used by "spectrum").

    "dominant"      -- the winning emotion's score (0..1): height = intensity,
                       and the color tells you of what.
    an emotion name -- that one emotion's score (0..1): a single clean arc
                       (e.g. --z-mode sadness for the sadness arc).
    "spectrum"      -- the emotions laid out in a fixed mood order (EMOTION_
                       SPECTRUM, dark->light) and the height is the score-
                       weighted position of the window's emotional blend on that
                       line (0..1). The axis ticks are the emotion names, so the
                       height reads directly as "which emotion-region is this".
    "pc1"           -- first principal component of the six emotion signals: the
                       single axis explaining the most emotional variation in
                       THIS book, normalized to [-1, 1] and oriented so its
                       strongest-loading emotion is the positive pole. The
                       loadings (printed) say what the poles mean.
  """
  if z_mode == "dominant":
    return pooled.max(axis=1), "dominant-emotion score", (0.0, 1.0), (0.0, 1.0), None

  if z_mode in emotions:
    return (pooled[:, emotions.index(z_mode)], f"{z_mode} score",
            (0.0, 1.0), (0.0, 1.0), None)

  if z_mode == "spectrum":
    missing = [e for e in emotions if e not in EMOTION_SPECTRUM]
    if missing:
      raise ValueError(
        f"EMOTION_SPECTRUM has no position for {missing}; it lists "
        f"{EMOTION_SPECTRUM}."
      )
    # Each emotion's fixed slot on the [0,1] line, then the window's height is
    # the score-weighted mean slot -- the emotional centre of mass on the
    # spectrum. Normalizing by the total score means only the *balance* of
    # emotions sets the position, not their raw intensity.
    n = len(EMOTION_SPECTRUM)
    pos_of = {e: i / (n - 1) for i, e in enumerate(EMOTION_SPECTRUM)}
    positions = np.array([pos_of[e] for e in emotions])   # aligned to columns
    denom = pooled.sum(axis=1)
    z = (pooled * positions).sum(axis=1) / np.where(denom > 1e-9, denom, 1.0)
    ticks = ([pos_of[e] for e in EMOTION_SPECTRUM], list(EMOTION_SPECTRUM))
    return z, "emotion spectrum (dark -> light)", (0.0, 1.0), (0.0, 1.0), ticks

  if z_mode == "pc1":
    pca = PCA(n_components=1)
    z = pca.fit_transform(pooled)[:, 0]
    loadings = pca.components_[0]
    # PCA's sign is arbitrary; pin the strongest-loading emotion to the positive
    # pole so the axis means the same thing on every run and every book.
    if loadings[int(np.argmax(np.abs(loadings)))] < 0:
      z, loadings = -z, -loadings
    scale = float(np.max(np.abs(z))) or 1.0
    z = z / scale                                    # -> [-1, 1], per book
    order = np.argsort(loadings)[::-1]
    pos, neg = emotions[order[0]], emotions[order[-1]]
    print(f"  PC1: {pca.explained_variance_ratio_[0]:.1%} of emotion variance")
    print("  loadings: " + ", ".join(
      f"{emotions[i]} {loadings[i]:+.2f}" for i in order))
    return z, f"emotion PC1 (+{pos} / -{neg})", (-1.0, 1.0), (-1.0, 1.0), None

  raise ValueError(
    f"--z-mode must be 'dominant', 'spectrum', 'pc1', or an emotion name "
    f"({', '.join(emotions)}); got {z_mode!r}"
  )


def emotion_terrain(coords, height, smoother, seed):
  """Smoothed height surface over the window plane, the grid-bandwidth way.

  Fits z = f(x, y) to the windows' heights with one of geometry/smoothers.py's
  estimators, exactly as bandwidth_grid.py / surface/kernel/loo.py do for the
  per-paragraph scores: standardize the coordinates so hx/hy are comparable,
  cross-validate the bandwidth on the same ladder, then evaluate on a grid masked
  where the kernel density is too thin to trust. Returns (GX, GY, Z, params) in
  the original coordinate units.

  Windows are few, so cross-validation is leave-one-out (folds = n), which is
  steadier here than a handful of k-folds. local_linear is the default because
  its tilted local fit follows the trend off the edge of the data instead of
  flattening it -- the boundary bias a plain kernel average suffers, and where
  the arc's opening and ending most often sit.
  """
  predict, _ = SMOOTHERS[smoother]
  grid = TERRAIN_GRIDS[smoother]

  xs = Standardizer().fit(coords)
  Xs = xs.forward(coords)
  params, _ = tune(predict, grid, Xs, height, folds=len(Xs), seed=seed)

  gx, gy, Z, mask, _ = fit_grid(predict, params, Xs, height,
                                resolution=120, margin=0.05,
                                density_floor_pct=5.0)
  raw = xs.inverse(np.column_stack([gx.ravel(), gy.ravel()]))
  GX, GY = raw[:, 0].reshape(gx.shape), raw[:, 1].reshape(gy.shape)
  tuned = "  ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}"
                    for k, v in params.items())
  print(f"  terrain[{smoother}]: {tuned}; "
        f"{int(mask.sum())} of {mask.size} cells above floor")
  return GX, GY, Z, params


def draw_arc_3d(ax, coords, height, codes, categories, terrain, clim,
                zlabel, zlim, zticks, marker_size=28):
  """Draw one 3-D arc (surface + path + colored windows + O/X) into `ax`.

  Shared by the standalone plot and the per-emotion facet panels so they agree
  down to the marker; only the framing (labels, legend, saving) differs.
  """
  if terrain is not None:
    gx, gy, Z, _ = terrain
    ax.plot_surface(gx, gy, Z, cmap=CMAP, vmin=clim[0], vmax=clim[1], linewidth=0,
                    antialiased=True, alpha=0.5, rstride=2, cstride=2, zorder=1)

  # The path in reading order, a neutral thread; color is carried by the marks.
  ax.plot(coords[:, 0], coords[:, 1], height, color="0.5", linewidth=0.8,
          alpha=0.7, zorder=2)

  for code, (label, hue, marker, _symbol) in enumerate(categories):
    mask = codes == code
    if not mask.any():
      continue
    recessive = label == UNCLEAR
    ax.scatter(coords[mask, 0], coords[mask, 1], height[mask],
               c=hue, marker=marker, s=marker_size * (0.7 if recessive else 1.0),
               alpha=0.4 if recessive else 0.95,
               edgecolors="none" if recessive else "#33322e",
               linewidths=0 if recessive else 0.3, depthshade=False,
               zorder=3, label=f"{label} ({int(mask.sum())})")

  ax.scatter(coords[0, 0], coords[0, 1], height[0], s=120, facecolor="none",
             edgecolor="black", linewidths=1.6, marker="o", zorder=4)
  ax.scatter(coords[-1, 0], coords[-1, 1], height[-1], s=140, color="black",
             marker="X", zorder=4)

  # Spectrum ticks are whole words; pad the axis title out so it clears them.
  ax.set_zlabel(zlabel, labelpad=32 if zticks is not None else 6)
  ax.set_zlim(*zlim)
  if zticks is not None:
    ax.set_zticks(zticks[0])
    ax.set_zticklabels(zticks[1])


def static_plot(coords, height, codes, categories, terrain, title,
                xlabel, ylabel, zlabel, zlim, clim, zticks, output_path):
  fig = plt.figure(figsize=(10, 8))
  ax = fig.add_subplot(111, projection="3d")
  draw_arc_3d(ax, coords, height, codes, categories, terrain, clim,
              zlabel, zlim, zticks)
  ax.set_xlabel(xlabel)
  ax.set_ylabel(ylabel)
  ax.set_title(title)
  ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
  fig.tight_layout()
  fig.savefig(output_path, dpi=200)
  plt.close(fig)
  print(f"  wrote {output_path}")


def geodesic_surface(coords, height, hx=0.2, hy=0.2, resolution=90, margin=0.05):
  """Fit one emotion's surface the way the geodesic work needs it.

  gaussian_nw at a small fixed bandwidth, the smoother geodesic_arrows.py insists
  on: it is differentiable everywhere, leaves no holes in the mesh, and cannot
  leave the [0, 1] score range -- all required for a clean mesh the exact solver
  can walk. Fitted in standardized coords, returned in raw PCA units.

  No density mask: the exact MMP solver's verify() asserts the mesh is a *single*
  connected component, but with only a few dozen windows a masked surface breaks
  into islands and the solver aborts. gaussian_nw is defined everywhere, so we
  keep the full rectangular grid (mask all True) -- one clean manifold. The
  far-field cells are a gentle extrapolation the reading-order path rarely visits.
  """
  xs = Standardizer().fit(coords)
  Xs = xs.forward(coords)

  lo, hi = Xs.min(axis=0), Xs.max(axis=0)
  pad = (hi - lo) * margin
  lo, hi = lo - pad, hi + pad
  gxs = np.linspace(lo[0], hi[0], resolution)
  gys = np.linspace(lo[1], hi[1], resolution)
  GXs, GYs = np.meshgrid(gxs, gys)
  Q = np.column_stack([GXs.ravel(), GYs.ravel()])

  values, _ = gaussian_nw(Xs, height, Q, hx, hy)
  Z = values.reshape(GXs.shape)
  mask = np.ones_like(Z, dtype=bool)

  raw = xs.inverse(Q)
  GX, GY = raw[:, 0].reshape(GXs.shape), raw[:, 1].reshape(GYs.shape)
  return GX, GY, Z, mask


def reading_order_geodesics(GX, GY, Z, mask, coords, alpha):
  """Trace the reading-order path as consecutive geodesics along the surface.

  Mesh the surface at vertical exaggeration `alpha`, snap each window to its foot
  on the mesh, then solve the exact geodesic for every consecutive window pair
  i -> i+1. The concatenation is the narrative path draped on the emotion
  landscape: where a hill sits between two windows the path bends around it
  instead of crossing straight. Returns the per-pair polylines (some may be None
  if the masked surface is disconnected there).
  """
  vertices, faces, _ = height_mesh(GX, GY, Z, mask, alpha)
  graph = edge_graph(vertices, faces)
  feet, _ = snap(coords, vertices)
  pairs = [(int(feet[i]), int(feet[i + 1])) for i in range(len(coords) - 1)]
  return exact_paths(vertices, faces, pairs, graph=graph)


def draw_geodesic_facet(ax, GX, GY, Z, paths, coords, height, alpha):
  """Surface + the reading-order geodesic curve, colored start->end. No markers."""
  ax.plot_surface(GX, GY, Z, cmap=CMAP, vmin=0.0, vmax=1.0, linewidth=0,
                  antialiased=True, alpha=0.5, rstride=2, cstride=2, zorder=1)

  cmap = plt.get_cmap("plasma")
  n = len(coords)
  drawn = 0
  for i, p in enumerate(paths):
    if p is None or len(p) < 2:
      continue
    # Undo the exaggeration so the curve sits on the plotted (score-unit) surface.
    z = p[:, 2] / alpha if alpha > 0 else np.zeros(len(p))
    ax.plot(p[:, 0], p[:, 1], z, color=cmap(i / max(n - 2, 1)),
            linewidth=1.6, alpha=0.95, zorder=3)
    drawn += 1

  # Opening (O) and ending (X) on the surface, at the windows' own heights.
  ax.scatter(coords[0, 0], coords[0, 1], height[0], s=90, facecolor="none",
             edgecolor="black", linewidths=1.6, marker="o", zorder=4)
  ax.scatter(coords[-1, 0], coords[-1, 1], height[-1], s=110, color="black",
             marker="X", zorder=4)
  ax.set_zlim(0.0, 1.0)
  return drawn


def build_facet_panels(coords, emo_pooled, emotions, smoother, seed, path_style,
                       alpha):
  """Per-emotion render data, computed ONCE so the PNG and HTML share it.

  The geodesic solve is the expensive step; doing it here means the static and
  interactive facets do not each pay for it. Returns a list of dicts, one per
  emotion, tagged by `path_style`.
  """
  panels = []
  for i, emotion in enumerate(emotions):
    height = emo_pooled[:, i]
    if path_style == "geodesic":
      GX, GY, Z, mask = geodesic_surface(coords, height)
      paths = reading_order_geodesics(GX, GY, Z, mask, coords, alpha)
      drawn = sum(p is not None and len(p) >= 2 for p in paths)
      print(f"    {emotion}: {drawn}/{len(coords) - 1} geodesic segments")
      panels.append(dict(emotion=emotion, height=height,
                         GX=GX, GY=GY, Z=Z, paths=paths))
    else:
      terrain = (None if smoother == "none"
                 else emotion_terrain(coords, height, smoother, seed))
      panels.append(dict(emotion=emotion, height=height, terrain=terrain))
  return panels


def facet_plot(panels, coords, codes, categories, xlabel, ylabel, title,
               output_path, path_style, alpha):
  """Static 2x3 grid of per-emotion arcs from precomputed `panels`.

  Only the height changes across panels, so scanning the grid shows where in the
  book each emotion rises and falls. `path_style` chooses what rides the surface:
    "markers"  -- the windows as dominant-emotion colored marks (+ terrain);
    "geodesic" -- the reading-order path as a curve drawn *along* the surface,
                  colored start->end, no markers (temporal ordering as geodesics).
  """
  n = len(panels)
  cols = 3
  rows = (n + cols - 1) // cols
  fig = plt.figure(figsize=(5.0 * cols, 4.4 * rows))

  for i, panel in enumerate(panels):
    ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
    if path_style == "geodesic":
      draw_geodesic_facet(ax, panel["GX"], panel["GY"], panel["Z"],
                          panel["paths"], coords, panel["height"], alpha)
      ax.set_zlabel(panel["emotion"], labelpad=6)
    else:
      draw_arc_3d(ax, coords, panel["height"], codes, categories,
                  panel["terrain"], (0.0, 1.0), panel["emotion"], (0.0, 1.0),
                  None, marker_size=16)
    ax.set_title(panel["emotion"], fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=7)
    ax.set_ylabel(ylabel, fontsize=7)
    ax.tick_params(labelsize=6)

  if path_style == "geodesic":
    sm = plt.cm.ScalarMappable(cmap="plasma",
                               norm=plt.Normalize(vmin=0.0, vmax=1.0))
    cbar = fig.colorbar(sm, ax=fig.axes, fraction=0.015, pad=0.02)
    cbar.set_label("reading order (start -> end)")
  else:
    handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=9,
               title="dominant emotion", framealpha=0.9)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

  fig.suptitle(title, fontsize=13)
  fig.savefig(output_path, dpi=170)
  plt.close(fig)
  print(f"  wrote {output_path}")


def _geodesic_polyline(paths, coords, alpha):
  """Flatten the per-pair geodesics into one gap-broken polyline + progression.

  Consecutive geodesics share an endpoint (pair i ends where pair i+1 begins), so
  concatenating them is a single continuous line; a None (missing pair) inserts a
  break so plotly does not draw across the gap. The per-vertex progression array
  drives the start->end color.
  """
  n = len(coords)
  xs, ys, zs, prog = [], [], [], []
  for i, p in enumerate(paths):
    if p is None or len(p) < 2:
      continue
    z = p[:, 2] / alpha if alpha > 0 else np.zeros(len(p))
    t = i / max(n - 2, 1)
    xs.extend(p[:, 0]); ys.extend(p[:, 1]); zs.extend(z); prog.extend([t] * len(p))
    xs.append(None); ys.append(None); zs.append(None); prog.append(t)
  return xs, ys, zs, prog


def interactive_facet_plot(panels, coords, categories, xlabel, ylabel, title,
                           output_path, path_style):
  """Rotatable 2x3 grid of the per-emotion arcs, one plotly scene per emotion."""
  n = len(panels)
  cols = 3
  rows = (n + cols - 1) // cols
  fig = make_subplots(
    rows=rows, cols=cols, specs=[[{"type": "scene"}] * cols for _ in range(rows)],
    subplot_titles=[p["emotion"] for p in panels], horizontal_spacing=0.02,
    vertical_spacing=0.06,
  )

  for i, panel in enumerate(panels):
    r, c = i // cols + 1, i % cols + 1
    GX, GY, Z = panel["GX"], panel["GY"], panel["Z"]
    fig.add_trace(go.Surface(
      x=GX[0], y=GY[:, 0], z=Z, colorscale="Viridis", cmin=0.0, cmax=1.0,
      opacity=0.55, showscale=False, hoverinfo="skip", name=panel["emotion"],
    ), row=r, col=c)

    xs, ys, zs, prog = _geodesic_polyline(panel["paths"], coords, 1.0)
    fig.add_trace(go.Scatter3d(
      x=xs, y=ys, z=zs, mode="lines",
      line=dict(color=prog, colorscale="Plasma", width=5, cmin=0.0, cmax=1.0),
      connectgaps=False, hoverinfo="skip", showlegend=False,
    ), row=r, col=c)

    for idx, sym in [(0, "circle-open"), (-1, "x")]:
      fig.add_trace(go.Scatter3d(
        x=[coords[idx, 0]], y=[coords[idx, 1]], z=[panel["height"][idx]],
        mode="markers", marker=dict(size=5, color="black", symbol=sym),
        hoverinfo="skip", showlegend=False,
      ), row=r, col=c)

  # Every scene shares the same axes: score height in [0,1], labelled x/y.
  scene = dict(xaxis_title=xlabel, yaxis_title=ylabel, zaxis_title="score",
               zaxis=dict(range=[0, 1]))
  fig.update_layout(title=title, margin=dict(l=0, r=0, t=60, b=0),
                    **{f"scene{i + 1 if i else ''}": scene for i in range(n)})
  fig.write_html(output_path, include_plotlyjs="cdn")
  print(f"  wrote {output_path}")


def interactive_plot(coords, height, codes, categories, terrain, hover, title,
                     xlabel, ylabel, zlabel, zlim, clim, zticks, output_path):
  traces = []

  if terrain is not None:
    gx, gy, Z, _ = terrain
    traces.append(go.Surface(
      x=gx[0], y=gy[:, 0], z=Z, colorscale="Viridis", cmin=clim[0], cmax=clim[1],
      opacity=0.5, showscale=False, hoverinfo="skip", name="emotion terrain",
    ))

  # The reading-order thread, so direction is legible when rotating.
  traces.append(go.Scatter3d(
    x=coords[:, 0], y=coords[:, 1], z=height, mode="lines",
    line=dict(color="rgba(120,120,120,0.7)", width=3),
    hoverinfo="skip", showlegend=False,
  ))

  hover = np.asarray(hover)
  for code, (label, hue, _marker, symbol) in enumerate(categories):
    mask = codes == code
    if not mask.any():
      continue
    recessive = label == UNCLEAR
    traces.append(go.Scatter3d(
      x=coords[mask, 0], y=coords[mask, 1], z=height[mask], mode="markers",
      name=f"{label} ({int(mask.sum())})",
      marker=dict(size=3 if recessive else 4.5, color=hue, symbol=symbol,
                  opacity=0.4 if recessive else 0.95,
                  line=dict(width=0 if recessive else 0.3, color="#33322e")),
      text=hover[mask], hoverinfo="text",
    ))

  # Start (O) and end (X) as their own labelled marks.
  for idx, name, sym in [(0, "start", "circle-open"), (-1, "end", "x")]:
    traces.append(go.Scatter3d(
      x=[coords[idx, 0]], y=[coords[idx, 1]], z=[height[idx]], mode="markers",
      name=name, marker=dict(size=7, color="black", symbol=sym),
      text=[hover[idx]], hoverinfo="text",
    ))

  zaxis = dict(range=list(zlim))
  if zticks is not None:
    zaxis.update(tickmode="array", tickvals=zticks[0], ticktext=zticks[1])

  fig = go.Figure(traces)
  fig.update_layout(
    title=title, legend=dict(title="dominant emotion"),
    scene=dict(xaxis_title=xlabel, yaxis_title=ylabel,
               zaxis_title=zlabel, zaxis=zaxis),
    margin=dict(l=0, r=0, t=40, b=0),
  )
  fig.write_html(output_path, include_plotlyjs="cdn")
  print(f"  wrote {output_path}")


def resolve_bandwidth(z_bandwidth):
  """CLI --z-bandwidth to the value dominant_windows wants: 'pool' -> None
  (boxcar), 'auto' -> "auto" (LOO-CV), else a float bandwidth in paragraphs.
  """
  if z_bandwidth == "pool":
    return None
  if z_bandwidth in ("auto", "loo"):
    return z_bandwidth
  try:
    return float(z_bandwidth)
  except ValueError:
    raise ValueError(
      f"--z-bandwidth must be 'pool', 'auto', 'loo', or a number, "
      f"got {z_bandwidth!r}"
    )


def run_model(book, model, args):
  print(f"\n=== {book} / {model} ===")

  paragraphs = load_paragraphs(book)
  embeddings = np.load(
    os.path.join("books", book, "embeddings", model, "embeddings.npy")
  )
  if len(embeddings) != len(paragraphs):
    raise ValueError(
      f"embeddings ({len(embeddings)}) and paragraphs ({len(paragraphs)}) mismatch."
    )

  windows = window_bounds(len(paragraphs), args.size, args.stride)
  print(f"  {len(paragraphs)} paragraphs -> {len(windows)} windows "
        f"(size {args.size}, stride {args.stride})")

  pooled = pool_embeddings(embeddings, windows, args.l2)
  bandwidth = resolve_bandwidth(args.z_bandwidth)
  emotions, emo_pooled, codes, categories = window_emotions(
    book, paragraphs, windows, args.min_score, bandwidth)

  facets = args.z_mode == "facets"
  if not facets:
    height, zlabel, zlim, clim, zticks = z_axis(emotions, emo_pooled, args.z_mode)
    labels = [c[0] for c in categories]
    # Hover: what each window is, its dominant emotion (the color) and its height.
    hover = [
      f"window {i} (paras {s}-{e - 1})<br>dominant: {labels[codes[i]]}"
      f"<br>{zlabel}: {height[i]:.2f}"
      for i, (s, e) in enumerate(windows)
    ]

  # No variant: win_tag/proj_tag below already put the window, stride, z-mode and
  # z-bandwidth into every filename.
  output_dir = paths.out_dir(
    book, model,
    os.path.join(paths.NARRATIVE_ARC_3D, paths.ARC_FROM_TEXT))
  paths.stamp(output_dir, __file__, args, book=book, model=model,
              emotions=emotions, z_bandwidth=bandwidth)
  win_tag =(f"w{args.size}_s{args.stride}" + ("_l2" if args.l2 else "")
             + f"_zb{args.z_bandwidth}")

  for proj in args.proj:
    print(f"  [{proj}]")
    coords, xlabel, ylabel, proj_tag = project(pooled, proj, args)
    base = os.path.join(output_dir,
                        f"arc3d_{proj}_{args.z_mode}_{win_tag}{proj_tag}")

    # PCA is the only genuine height field -- two windows can share a UMAP/t-SNE
    # (x, y) at different heights, so a surface over those is a smoothing, not a
    # true landscape. We still fit it on any projection when asked (the
    # window-60/stride-30 UMAP view reads well), but say so honestly.
    if args.terrain_smoother != "none" and proj != "pca":
      print(f"  note: {proj} base is non-metric; terrain is a smoothing, "
            f"not a metric landscape")

    if facets:
      # One panel per emotion; compute each panel's surface/geodesics once, then
      # render both the static grid and (for geodesics) the rotatable HTML.
      style = {"markers": "z=score", "geodesic": "reading-order geodesic"}[
        args.facet_path]
      title = (f"Narrative arc emotions ({style}) -- {proj.upper()} base "
               f"(window {args.size}, stride {args.stride})")
      facet_base = f"{base}_{args.facet_path}"
      panels = build_facet_panels(coords, emo_pooled, emotions,
                                  args.terrain_smoother, args.seed,
                                  args.facet_path, args.facet_alpha)
      facet_plot(panels, coords, codes, categories, xlabel, ylabel, title,
                 facet_base + ".png", args.facet_path, args.facet_alpha)
      if args.facet_path == "geodesic" and HAS_PLOTLY:
        interactive_facet_plot(panels, coords, categories, xlabel, ylabel, title,
                               facet_base + ".html", args.facet_path)
      continue

    terrain = (None if args.terrain_smoother == "none"
               else emotion_terrain(coords, height, args.terrain_smoother,
                                    args.seed))
    title = (f"Narrative arc in 3-D -- {proj.upper()} base, z = {zlabel}\n"
             f"window {args.size}, stride {args.stride}")
    static_plot(coords, height, codes, categories, terrain, title,
                xlabel, ylabel, zlabel, zlim, clim, zticks, base + ".png")
    if HAS_PLOTLY:
      interactive_plot(coords, height, codes, categories, terrain, hover, title,
                       xlabel, ylabel, zlabel, zlim, clim, zticks, base + ".html")

  if not facets and not HAS_PLOTLY:
    print("  plotly not installed -> only PNGs written (pip install plotly).")
  print(f"  -> {output_dir}")


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--book", default="alice_wonderland", type=str)
  parser.add_argument("--model", type=str)
  parser.add_argument("--all-models", action="store_true")
  parser.add_argument("--size", default=25, type=int, help="Window length in paragraphs.")
  parser.add_argument("--stride", default=12, type=int, help="Paragraphs between windows.")
  parser.add_argument("--l2", action="store_true",
                      help="L2-normalize embeddings before pooling.")
  parser.add_argument("--proj", nargs="+", default=["pca", "umap", "tsne"],
                      choices=["pca", "umap", "tsne"],
                      help="Base projections for the x-y plane.")
  parser.add_argument("--z-mode", default="dominant",
                      help="What the height axis carries: 'dominant' (winning "
                           "emotion's score, 0..1); 'spectrum' (emotions laid "
                           "out dark->light, height = the blend's weighted "
                           "position on that labelled line, 0..1); 'pc1' (first "
                           "principal component of the six emotions, the book's "
                           "main axis of emotional variation, [-1,1]); or a "
                           "single emotion name (that emotion's score, 0..1); "
                           "or 'facets' for a 2x3 grid of all six emotions, one "
                           "per panel. Marker color is always the dominant "
                           "emotion.")
  parser.add_argument("--z-bandwidth", default="auto",
                      help="Emotion-height smoothing over reading position. "
                           "'auto' = Gaussian at sigma=window/2 (smooth analog "
                           "of the box); 'loo' = data-driven LOO-CV bandwidth "
                           "(can be rougher); a number sets sigma directly (in "
                           "paragraphs); 'pool' restores the boxcar height.")
  parser.add_argument("--facet-path", default="markers",
                      choices=["markers", "geodesic"],
                      help="For --z-mode facets: 'markers' draws the windows as "
                           "dominant-emotion marks; 'geodesic' instead traces the "
                           "reading-order path as a curve along each emotion's "
                           "surface (exact geodesics), colored start->end, no "
                           "markers.")
  parser.add_argument("--facet-alpha", default=1.0, type=float,
                      help="Vertical exaggeration of the surface for "
                           "--facet-path geodesic. Larger bends the path more "
                           "around the emotion's hills.")
  parser.add_argument("--terrain-smoother", default="local_linear",
                      choices=["local_linear", "gaussian_nw", "epanechnikov_nw",
                               "loess", "none"],
                      help="Smoother for the height surface, fit the "
                           "grid-bandwidth way (standardized coords, CV-tuned "
                           "bandwidth, density-masked). 'local_linear' corrects "
                           "the edge-flattening of a plain kernel average; "
                           "'none' draws the arc without a surface. Fitted on "
                           "every requested projection, not just PCA.")
  parser.add_argument("--min-score", default=0.2, type=float,
                      help="Below this pooled top score a window is 'unclear'.")
  parser.add_argument("--seed", default=0, type=int)
  parser.add_argument("--neighbors", default=15, type=int, help="UMAP n_neighbors.")
  parser.add_argument("--min-dist", default=0.1, type=float, help="UMAP min_dist.")
  parser.add_argument("--perplexity", default=30.0, type=float,
                      help="t-SNE perplexity; clamped below the window count.")
  args = parser.parse_args()

  if args.stride < 1:
    parser.error("--stride must be >= 1")

  if args.all_models:
    models = sorted(os.listdir(os.path.join("books", args.book, "embeddings")))
  elif args.model:
    models = [args.model]
  else:
    parser.error("pass --model or --all-models")

  for model in models:
    run_model(args.book, model, args)

  print("\nDone.")


if __name__ == "__main__":
  main()
