"""
The narrative arc lifted into 3-D: emotion is the height.

The 2-D arc (arc/curve/arc_2d.py) draws the sliding-window path on a flat
projection, where x and y are arbitrary UMAP/t-SNE (or PCA) coordinates with no
units. Here the third axis carries a real, metric quantity so the story becomes
a curve rising and falling over the semantic map. What that quantity is is up
to --z-mode:

  spectrum (default)  the emotions laid out in a fixed mood order (dark->light);
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
collapse. The marker color is always reading order (plasma, start -> end).

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

Each projection writes a static PNG. Everything lands in arc_on_surface/output/<book>/<model>/narrative_arc_3d/.

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
from data import load_paragraphs, load_scores
from windows import pool as pool_embeddings, window_bounds, window_centers
from geometry.scalar_field import (
  bandwidth_candidates, loo_bandwidth, nadaraya_watson,
)
from geometry.smoothers import SMOOTHERS
from smooth_common import Standardizer, fit_grid, tune
from surface.mood.mood import SPECTRUM
from surface.kernel.gaussian import GRID as GAUSSIAN_GRID
from surface.kernel.epanechnikov import GRID as EPANECHNIKOV_GRID
from surface.kernel.local_linear import GRID as LOCAL_LINEAR_GRID
from surface.kernel.loess import GRID as LOESS_GRID

CMAP = "viridis"
ORDER_CMAP = "plasma"   # the windows' marker color: reading order, start -> end


def progression(n):
  """Each of n windows' place in the book, 0 at the first and 1 at the last."""
  return np.linspace(0.0, 1.0, n)

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


def window_emotions(book, paragraphs, windows, bandwidth):
  """Smoothed per-window emotion matrix.

  `bandwidth` chooses how each emotion is read at each window:
    None    -- block-mean pool over the window (the original step-like boxcar);
    "auto"  -- Gaussian over reading position at sigma = window/2, a smooth
               analog of the box at the same scale (the sensible default);
    "loo"   -- Gaussian at the leave-one-out-CV bandwidth (data-driven, can be
               narrower than the window and so *rougher* -- see the caveat above);
    a float -- Gaussian at that bandwidth in paragraphs.

  Returns (emotions, pooled).
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

  return emotions, pooled


def z_axis(emotions, pooled, z_mode):
  """The height each window sits at, and how to label and scale that axis.

  The six emotions are six numbers per window; the z-axis is one. This chooses
  how to collapse them -- and so what reading the height actually tells you.
  Marker *color* is separate (always reading order), so only height's
  meaning changes here. Returns (height, label, zlim, clim, zticks): zlim the
  axis range, clim the terrain colormap range, and zticks either None or a
  (positions, labels) pair to label the axis with (used by "spectrum").

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
    f"--z-mode must be 'spectrum', 'pc1', or an emotion name "
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


def draw_arc_3d(ax, coords, height, terrain, clim, zlabel, zlim, zticks,
                marker_size=28):
  """Draw one 3-D arc (surface + path + colored windows + O/X) into `ax`."""
  if terrain is not None:
    gx, gy, Z, _ = terrain
    ax.plot_surface(gx, gy, Z, cmap=CMAP, vmin=clim[0], vmax=clim[1], linewidth=0,
                    antialiased=True, alpha=0.5, rstride=2, cstride=2, zorder=1)

  # The path in reading order, a neutral thread; the marks carry the order.
  ax.plot(coords[:, 0], coords[:, 1], height, color="0.5", linewidth=0.8,
          alpha=0.7, zorder=2)
  ax.scatter(coords[:, 0], coords[:, 1], height, c=progression(len(coords)),
             cmap=ORDER_CMAP, vmin=0.0, vmax=1.0, s=marker_size, alpha=0.95,
             edgecolors="#33322e", linewidths=0.3, depthshade=False, zorder=3)

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


def order_colorbar(fig, axes):
  sm = plt.cm.ScalarMappable(cmap=ORDER_CMAP, norm=plt.Normalize(vmin=0.0, vmax=1.0))
  cbar = fig.colorbar(sm, ax=axes, fraction=0.015, pad=0.12, shrink=0.6)
  cbar.set_label("reading order (start -> end)")


def static_plot(coords, height, terrain, title, xlabel, ylabel, zlabel, zlim,
                clim, zticks, output_path):
  fig = plt.figure(figsize=(10, 8))
  ax = fig.add_subplot(111, projection="3d")
  draw_arc_3d(ax, coords, height, terrain, clim, zlabel, zlim, zticks)
  ax.set_xlabel(xlabel)
  ax.set_ylabel(ylabel)
  ax.set_title(title)
  order_colorbar(fig, ax)
  fig.tight_layout()
  fig.savefig(output_path, dpi=200)
  plt.close(fig)
  print(f"  wrote {output_path}")


def resolve_bandwidth(z_bandwidth):
  """CLI --z-bandwidth to the value window_emotions wants: 'pool' -> None
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
  emotions, emo_pooled = window_emotions(book, paragraphs, windows, bandwidth)

  height, zlabel, zlim, clim, zticks = z_axis(emotions, emo_pooled, args.z_mode)

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

    terrain = (None if args.terrain_smoother == "none"
               else emotion_terrain(coords, height, args.terrain_smoother,
                                    args.seed))
    title = (f"Narrative arc in 3-D -- {proj.upper()} base, z = {zlabel}\n"
             f"window {args.size}, stride {args.stride}")
    static_plot(coords, height, terrain, title, xlabel, ylabel, zlabel, zlim,
                clim, zticks, base + ".png")

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
  parser.add_argument("--z-mode", default="spectrum",
                      help="What the height axis carries: 'spectrum' (emotions laid "
                           "out dark->light, height = the blend's weighted "
                           "position on that labelled line, 0..1); 'pc1' (first "
                           "principal component of the six emotions, the book's "
                           "main axis of emotional variation, [-1,1]); or a "
                           "single emotion name (that emotion's score, 0..1). "
                           "Marker color is always reading order.")
  parser.add_argument("--z-bandwidth", default="auto",
                      help="Emotion-height smoothing over reading position. "
                           "'auto' = Gaussian at sigma=window/2 (smooth analog "
                           "of the box); 'loo' = data-driven LOO-CV bandwidth "
                           "(can be rougher); a number sets sigma directly (in "
                           "paragraphs); 'pool' restores the boxcar height.")
  parser.add_argument("--terrain-smoother", default="local_linear",
                      choices=["local_linear", "gaussian_nw", "epanechnikov_nw",
                               "loess", "none"],
                      help="Smoother for the height surface, fit the "
                           "grid-bandwidth way (standardized coords, CV-tuned "
                           "bandwidth, density-masked). 'local_linear' corrects "
                           "the edge-flattening of a plain kernel average; "
                           "'none' draws the arc without a surface. Fitted on "
                           "every requested projection, not just PCA.")
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
