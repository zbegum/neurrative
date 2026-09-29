"""
One mood surface, fitted to the paragraphs themselves.

`arc_emotion_axis.py` builds six emotion surfaces and blends them cell by cell.
This does the two steps in the other order: collapse each paragraph's six scores
to a single mood number first, then fit **one** Gaussian surface to those 789
numbers. The result is one landscape with one z axis, ticked in emotion names.

    mood(p) = sum_e position(e) * w_e(p)      w = softmax(normalized score / temp)
    Z(x, y) = gaussian_nw over the paragraphs' (x, y, mood)

Why the order matters
---------------------
Blend-then-smooth and smooth-then-blend are not the same surface. Smoothing six
surfaces first averages each emotion over its neighbourhood before deciding which
one wins, so a lone wonder-heavy paragraph is diluted by its neighbours and may
never reach the top of the axis. Blending first lets that paragraph carry its own
mood into the fit, and the smoother then averages *moods*. This file exists to
make that comparison possible rather than to assert which is right.

It also means the surface is fitted to real per-paragraph values -- the same
thing every other surface in `surface/` is fitted to -- so the bandwidth work,
the density floor and the residual plots all mean here what they mean there.

The three knobs, and what each does to the picture
--------------------------------------------------
  --norm    how the six emotions are made comparable before blending. They are
            not comparable raw: some sit near 0.4 all book, some near 0, so an
            unnormalized blend is dominated by whichever is loudest on average
            and barely moves. `rank` (percentile within each emotion) spreads
            the most; `zscore` and `minmax` keep more of the original shape.
  --temp    softmax temperature. Small snaps each paragraph to its dominant
            emotion, so the mood is nearly categorical and the surface has
            plateaus with cliffs between them. Large averages the six, so every
            paragraph lands mid-spectrum and the surface flattens toward its mean.
  --hx/--hy kernel bandwidth, in standardized PCA units. The usual trade: small
            follows individual paragraphs (spiky), large is a broad swell.

`--sweep` renders a grid over temperature x bandwidth so the three can be judged
against each other instead of one at a time.

The spectrum order is an editorial choice, not something the data fixes; --order
sets what "up" means:

    sadness -> danger -> confusion -> curiosity -> wonder -> humor

Examples:

  python visualization/mood_surface.py --model bge-m3 --sweep
  python visualization/mood_surface.py --model bge-m3 --temp 0.3 --hx 0.15
  python visualization/mood_surface.py --model bge-m3 --sweep --norm zscore
"""

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import paths
from data import load_paragraphs, load_scores
from smooth_common import load_pca
from arc_on_surface import (build_surface, geodesic_route, resample,
                            surface_at, window_points)
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from arc_emotion_axis import DEFAULT_ORDER, normalizer, mood

CMAP = "magma"


def interactive(GX, GY, Z, X_raw, m, v, order, positions, chapters, summaries,
                emotions, matrix, args, out_dir, proj):
  """The mood surface as a rotatable page, with every paragraph inspectable.

  The static PNG can show the terrain or the points but never lets you ask which
  paragraph a bump is. Here each of the 789 points carries its reading position,
  its mood, its dominant emotion, all six scores and the annotator's one-line
  summary, so a feature of the surface can be traced back to the text that made
  it.

  Two colourings of the same points, one visible at a time:

    reading order    plasma over paragraph index -- where the book's beginning,
                     middle and end sit on the landscape. This is the question
                     the arc scripts answer with a curve; here it is answered
                     without assuming the points form a path at all.
    dominant emotion one colour per emotion -- the regions the mood surface is
                     smoothing over, before any smoothing.
  """
  fig = go.Figure()
  # Grey terrain: the height already carries the mood, and a warm surface would
  # swallow the points, which are the thing to inspect here.
  fig.add_surface(x=GX, y=GY, z=Z, colorscale="Greys", cmin=-0.15, cmax=1.15,
                  opacity=0.55, showscale=False, name="mood surface",
                  showlegend=True,
                  hovertemplate="PC1 %{x:.3f}<br>PC2 %{y:.3f}<br>"
                                "mood %{z:.3f}<extra>surface</extra>")

  win = np.argmax(v, axis=0)
  text = []
  for i in range(len(X_raw)):
    scores = "  ".join(f"{e} {matrix[i, emotions.index(e)]:.2f}" for e in order)
    text.append(f"paragraph {i}"
                + (f" &middot; chapter {chapters[i]}" if chapters is not None else "")
                + f"<br>mood {m[i]:.3f} &middot; dominant {order[win[i]]}"
                + f"<br>{scores}<br><br>{summaries[i]}")

  fig.add_scatter3d(
    x=X_raw[:, 0], y=X_raw[:, 1], z=m, mode="markers",
    marker=dict(size=3.4, color=np.arange(len(X_raw)), colorscale="Plasma",
                showscale=True, line=dict(width=0),
                colorbar=dict(title="reading<br>position", len=0.55, thickness=14)),
    name="paragraphs, by reading order", text=text, hoverinfo="text")

  for k, e in enumerate(order):
    sel = win == k
    if not sel.any():
      continue
    fig.add_scatter3d(
      x=X_raw[sel, 0], y=X_raw[sel, 1], z=m[sel], mode="markers",
      marker=dict(size=3.4, line=dict(width=0)),
      name=f"{e} ({int(sel.sum())})", visible="legendonly",
      text=[t for t, sl in zip(text, sel) if sl], hoverinfo="text")

  fig.update_layout(
    title=(f"The mood surface -- {args.book} / {args.model}<br>"
           f"<sub>blend {args.blend}, norm {args.norm}"
           + (f", temp {args.temp:g}" if args.blend == "softmax" else "")
           + f", h {args.hx:g} &middot; z = mood, sadness (low) to humor (high) "
             f"&middot; click a legend entry to colour by dominant emotion</sub>"),
    scene=dict(xaxis_title="PC1" if proj == "pca" else "UMAP-1",
               yaxis_title="PC2" if proj == "pca" else "UMAP-2",
               zaxis_title="mood",
               zaxis=dict(range=[0, 1], tickvals=list(positions),
                          ticktext=list(order)),
               aspectmode="cube"),
    legend=dict(orientation="h", y=-0.04),
    margin=dict(l=0, r=0, t=80, b=0), height=820)

  p_out = os.path.join(out_dir, paths.named("mood_surface", "interactive", "html", projection=proj))
  fig.write_html(p_out, include_plotlyjs="cdn")
  print(f"  wrote {p_out}")
  return p_out


def load_plane(book, model, plane, beta):
  """The 2-D plane the surface is fitted over, and a label for the filenames.

  pca         the linear projection every other surface in this repo uses.
  chain_umap  UMAP run over a kNN graph with reading-order edges added, at the
              chosen beta -- how much those temporal edges are worth. Higher
              beta pulls consecutive paragraphs together, so the layout trades
              neighbourhood fidelity (trust) for reading-order continuity
              (order); the sweep records both per beta.

  A caveat that belongs with the choice rather than repeated later: UMAP is not
  metric, so distances on this plane are not comparable and anything measured in
  them -- geodesic lengths, climb ratios -- means less than it does over PCA.
  The height itself is still well defined (each paragraph has one (x, y)), so
  the surface and its shape are readable; it is the *lengths* that are not.
  """
  if plane == "pca":
    return np.asarray(load_pca(book, model), dtype=np.float64)[:, :2], "pca"

  root = os.path.join(paths.ROOT, "output", book, model, paths.CHAIN_UMAP)
  if not os.path.isdir(root):
    raise SystemExit(f"no chain_umap runs under {root}")
  hits = []
  for run in sorted(os.listdir(root)):
    d = os.path.join(root, run)
    if not os.path.isdir(d):
      continue
    for f in sorted(os.listdir(d)):
      if f.startswith(f"chain_umap_b{beta:g}_") and f.endswith(".npy"):
        hits.append(os.path.join(d, f))
  if not hits:
    raise SystemExit(f"no chain_umap coordinates for beta={beta:g} under {root}")
  if len(hits) > 1:
    print(f"  ({len(hits)} runs have beta={beta:g}; using {os.path.basename(hits[0])})")
  return np.asarray(np.load(hits[0]), dtype=np.float64)[:, :2], f"chainumap-b{beta:g}"


def draw_route(ax, GX, GY, Z, P, keys, args, cell):
  """Join the sampled points along the surface, not through it.

  A straight segment between two points on a landscape is a tunnel: it leaves
  the surface as soon as the ground between them is not flat. The geodesic is
  the path that stays on it, so the drawn curve is something the terrain
  actually contains. Returns (length, note).

  `--alpha` decides what "shortest" means here, because it is the exchange rate
  between mood units and plane units; at alpha 0 the surface is flat and every
  geodesic is a straight line.
  """
  runs, st = geodesic_route(GX, GY, Z, P, args.alpha, args.snap_tol * cell)
  total = 0.0
  for Q, tt in runs:
    draw_arc(ax, Q[:, :2], Q[:, 2], np.interp(tt, np.arange(len(keys)), keys),
             args.draw_lift, args.route_width, (0, len(GX) and args.n_paragraphs - 1))
    total += float(np.linalg.norm(np.diff(Q, axis=0), axis=1).sum())
  return total, f"{st['legs_solved']}/{st['legs_total']} geodesic legs"


def dp_simplify(t, y, keep):
  """Douglas-Peucker on the mood-over-reading-position curve, down to `keep`.

  Returns the indices whose (reading position, mood) polyline best preserves the
  shape of the whole one. Unlike uniform sampling it spends its points where the
  emotional trajectory actually turns, and leaves long flat stretches to a
  single segment -- which is exactly the summary a narrative arc wants.
  """
  P = np.column_stack([(t - t.min()) / max(t.ptp(), 1e-9), y])
  chosen = {0, len(P) - 1}
  segs = [(0, len(P) - 1)]
  while len(chosen) < keep and segs:
    best = None
    for a, b in segs:
      if b - a < 2:
        continue
      d = P[b] - P[a]
      n = np.linalg.norm(d)
      if n == 0:
        dev = np.linalg.norm(P[a + 1:b] - P[a], axis=1)
      else:
        dev = np.abs(np.cross(d / n, P[a + 1:b] - P[a]))
      i = int(np.argmax(dev)) + a + 1
      if best is None or dev.max() > best[0]:
        best = (dev.max(), i, a, b)
    if best is None:
      break
    _, i, a, b = best
    chosen.add(i)
    segs = [x for x in segs if x != (a, b)] + [(a, i), (i, b)]
  return np.array(sorted(chosen))


def sample_paragraphs(strategy, X_raw, m, v, chapters, n_points, rng):
  """An ordered subset of paragraphs to draw the arc through.

  Every strategy returns indices in reading order -- the arc is still the story's
  sequence, only the sampling changes. What differs is what each spends its
  points on.
  """
  n = len(X_raw)
  t = np.arange(n, dtype=float)
  if strategy == "uniform":
    return np.unique(np.linspace(0, n - 1, n_points).astype(int))
  if strategy == "chapter medoid":
    out = []
    for c in np.unique(chapters):
      sel = np.where(chapters == c)[0]
      P = X_raw[sel]
      D = np.linalg.norm(P[:, None] - P[None], axis=2)
      out.append(sel[int(D.sum(1).argmin())])
    return np.array(sorted(out))
  if strategy == "mood extrema":
    # Turning points of the mood series: where the story changes emotional
    # direction, which is what a reader would call a beat.
    d = np.diff(m)
    turn = np.where(np.sign(d[:-1]) != np.sign(d[1:]))[0] + 1
    if len(turn) > n_points:
      prom = np.abs(m[turn] - np.median(m))
      turn = turn[np.argsort(-prom)[:n_points]]
    return np.array(sorted(set(turn.tolist()) | {0, n - 1}))
  if strategy == "douglas-peucker":
    return dp_simplify(t, m, n_points)
  if strategy == "decisive":
    srt = np.sort(v, axis=0)
    margin = srt[-1] - srt[-2]
    idx = np.argsort(-margin)[:n_points]
    return np.array(sorted(set(idx.tolist()) | {0, n - 1}))
  if strategy == "farthest point":
    # Maximal spatial spread: the arc visits every region of the terrain rather
    # than circling the crowded middle.
    sel = [int(np.argmax(np.linalg.norm(X_raw - X_raw.mean(0), axis=1)))]
    d = np.linalg.norm(X_raw - X_raw[sel[0]], axis=1)
    while len(sel) < n_points:
      i = int(np.argmax(d))
      sel.append(i)
      d = np.minimum(d, np.linalg.norm(X_raw - X_raw[i], axis=1))
    return np.array(sorted(set(sel) | {0, n - 1}))
  raise ValueError(f"unknown strategy {strategy!r}")


def narrative_units(chapters, target):
  """Split the book into ~`target` contiguous units, respecting chapter breaks.

  Chapters are the one segmentation of a novel that is not ours to invent, so
  the sweep refines or coarsens them rather than ignoring them: above the
  chapter count each chapter is divided into equal parts, below it whole
  chapters are grouped. Units are always contiguous in reading order, so the arc
  through their medoids is still the story's sequence.
  """
  ids = np.unique(chapters)
  runs = [np.where(chapters == c)[0] for c in ids]
  if target >= len(runs):
    per = max(1, int(round(target / len(runs))))
    return [part for r in runs for part in np.array_split(r, min(per, len(r)))]
  return [np.concatenate([runs[i] for i in grp])
          for grp in np.array_split(np.arange(len(runs)), target)]


def unit_medoids(X_raw, units):
  """The most central real paragraph in each unit."""
  out = []
  for u in units:
    P = X_raw[u]
    D = np.linalg.norm(P[:, None] - P[None], axis=2)
    out.append(int(u[int(D.sum(1).argmin())]))
  return np.array(sorted(out))


def bws_default(args):
  """Bandwidths to show when comparing blends: a narrow and a broad one."""
  return args.bandwidths[:2] if len(args.bandwidths) >= 2 else [args.hx]


def paragraph_mood(matrix, emotions, order, norm, temp, blend="softmax"):
  """One mood value per paragraph, by one of four ways of collapsing six.

  The problem this function is about: how do you put six emotions on one height
  without the height turning to mush? Three of the four avoid the averaging that
  does it.

    softmax   the weighted mean of spectrum positions, weights from a softmax of
              the normalized scores. Readable, but it is a *mean*: unless one
              emotion dominates, the answer is near the middle. Raising --temp
              averages harder, so the whole book converges on 0.5 and the
              surface flattens. This is the default and the thing to beat.

    project   the dot product of the normalized scores with the spectrum,
              centred so sadness pulls down and humor pulls up, with no division
              by the total. Nothing forces it toward the middle: a paragraph
              high in both wonder and humor goes higher still, and one high in
              nothing stays put. Intensity survives, which is what the mean
              throws away.

    dominant  the position of the winning emotion, full stop. Perfectly
              readable -- every height *is* a named emotion -- but a step
              function: no sense of how strongly, and cliffs between plateaus.

    pc1       the first principal component of the six normalized scores: the
              single axis the emotions themselves vary along most, rather than
              one chosen by hand. Sign-aligned to the spectrum so up stays
              positive. Says what the data thinks the axis is.

  Returns (mood, positions, normalized_matrix), mood scaled to [0, 1].
  """
  positions = np.linspace(0.0, 1.0, len(order))
  cols = [matrix[:, emotions.index(e)] for e in order]
  v = np.vstack([normalizer(c, norm)(c) for c in cols])          # (k, n)

  if blend == "softmax":
    return mood(v, positions, temp), positions, v

  if blend == "project":
    # Centred positions: -1 at sadness, +1 at humor. A sum, not a mean.
    signed = np.linspace(-1.0, 1.0, len(order))
    z = (signed.reshape(-1, 1) * v).sum(axis=0)
  elif blend == "dominant":
    z = positions[np.argmax(v, axis=0)]
  elif blend == "banded":
    # The fix for the real problem: a mean cannot separate six emotions on one
    # axis, and argmax separates them but throws away intensity. So give each
    # emotion its own band of the axis, centred on its spectrum position, and
    # use height *within* the band for how clearly it wins.
    #
    # The margin (winner minus runner-up) is ranked inside each emotion's own
    # group, so every band fills evenly instead of bunching wherever that
    # emotion's margins happen to sit. Bands are 90% of the spacing, so they
    # never overlap: a height still names exactly one emotion, but it also says
    # whether the paragraph is deep in that mood or on its edge.
    d = positions[1] - positions[0] if len(positions) > 1 else 1.0
    srt = np.sort(v, axis=0)
    win = np.argmax(v, axis=0)
    margin = srt[-1] - srt[-2]
    z = positions[win].astype(float)
    for k in range(len(positions)):
      sel = win == k
      if not sel.any():
        continue
      mk = margin[sel]
      q = (np.argsort(np.argsort(mk)) / max(len(mk) - 1, 1)) if len(mk) > 1 \
          else np.full(len(mk), 0.5)
      z[sel] += (q - 0.5) * d * 0.9
  elif blend == "pc1":
    A = (v - v.mean(axis=1, keepdims=True)).T                     # (n, k)
    _, _, Vt = np.linalg.svd(A, full_matrices=False)
    axis = Vt[0]
    if np.dot(axis, np.linspace(-1.0, 1.0, len(order))) < 0:
      axis = -axis                                                # up = positive
    z = A @ axis
  else:
    raise ValueError(f"unknown --blend {blend!r}")

  lo, hi = z.min(), z.max()
  return (z - lo) / (hi - lo if hi > lo else 1.0), positions, v


def spectrum_ticks(m, v, positions):
  """Where each emotion sits on *this* height axis, measured not assumed.

  Only `softmax` and `dominant` put a paragraph at its emotion's nominal
  position; `project` and `pc1` produce a height in their own units. So the tick
  for an emotion is the median height of the paragraphs it dominates -- the axis
  stays readable as a mood whatever the blend, and if a blend fails to separate
  the emotions the ticks bunch up and say so.
  """
  win = np.argmax(v, axis=0)
  ticks, labels = [], []
  for k in range(len(positions)):
    sel = win == k
    if sel.sum():
      ticks.append(float(np.median(m[sel])))
      labels.append(int(sel.sum()))
  return np.array(ticks), labels


def zticks(order, positions, ticks, counts, mode):
  """Tick positions and labels for the mood axis, without collisions.

  nominal   the spectrum itself: sadness 0.0 ... humor 1.0, evenly spaced. This
            is what the height *means*, so it always reads, and the labels can
            never overlap because the spacing is fixed.
  measured  where each emotion's paragraphs actually land. More informative --
            it shows which emotions the blend fails to separate -- but under a
            mean-based blend four of the six land on top of each other. Labels
            closer than a quarter of the nominal spacing are dropped, keeping
            the emotion with more paragraphs, and the dropped ones are reported
            rather than silently overplotted.
  """
  if mode == "nominal":
    return positions, list(order), []
  d = (positions[1] - positions[0]) if len(positions) > 1 else 1.0
  items = sorted(zip(ticks, order, counts), key=lambda r: -r[2])
  kept, dropped = [], []
  for t, e, c in items:
    if any(abs(t - k[0]) < 0.25 * d for k in kept):
      dropped.append(e)
    else:
      kept.append((t, e, c))
  kept.sort()
  return ([k[0] for k in kept], [f"{k[1]} ({k[2]})" for k in kept], dropped)


def window_settings(book, model):
  """Every (size, stride) `windows.py` has already saved, newest layout first.

  The window series are not recomputed here: `windows.py` is the one definition
  of what a window is, and it has written each setting's starts and stops to
  disk. Reading them means this grid cannot disagree with the rest of the repo
  about where window 12 begins.
  """
  root = os.path.join(paths.ROOT, "output", book, model, paths.WINDOWS)
  if not os.path.isdir(root):
    return []
  out = []
  for name in sorted(os.listdir(root)):
    f = os.path.join(root, name, "series.npz")
    if not os.path.exists(f):
      continue
    d = np.load(f, allow_pickle=True)
    meta = read_params(os.path.join(root, name, "params.json"))
    out.append({"name": name,
                "size": meta.get("size"), "stride": meta.get("stride"),
                "windows": list(zip(d["starts"].tolist(), d["stops"].tolist()))})
  out.sort(key=lambda r: (r["size"] or 0, r["stride"] or 0))
  return out


def read_params(path):
  try:
    with open(path) as f:
      return json.load(f)
  except OSError:
    return {}


ROUTE_CMAP = "winter"


def reading_bar(fig, n, label="reading position (paragraph)"):
  """One colourbar for the whole grid, saying what the route's colour means.

  Every panel shares the same scale -- 0 to the last paragraph -- so a colour is
  comparable across panels, which is the point of a grid. Drawn in its own axes
  under the panels rather than beside one of them, for the same reason.
  """
  sm = plt.cm.ScalarMappable(cmap=ROUTE_CMAP, norm=plt.Normalize(0, n - 1))
  sm.set_array([])
  cax = fig.add_axes([0.34, 0.045, 0.32, 0.012])
  cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
  cb.set_label(label, fontsize=9)
  cb.ax.tick_params(labelsize=8)
  return cb


def draw_arc(ax, arc_xy, z, centers, lift, lw=1.4, clim=None):
  """The route over the mood terrain, coloured by reading position."""
  pts = np.column_stack([arc_xy[:, 0], arc_xy[:, 1], z + lift])
  seg = np.stack([pts[:-1], pts[1:]], axis=1)
  # winter (blue -> green) against magma (purple -> orange): the route has to
  # read as not-terrain at a glance, and a warm ramp on a warm surface does not.
  lc = Line3DCollection(seg, cmap=ROUTE_CMAP, linewidths=lw, zorder=6)
  lc.set_array(np.asarray(centers[:-1], dtype=float))
  # Fixed limits, not per-panel autoscaling: otherwise the same colour means a
  # different paragraph in each panel and the shared bar would be a lie.
  lc.set_clim(*(clim if clim is not None else (float(centers[0]),
                                               float(centers[-1]))))
  ax.add_collection3d(lc, autolim=False)
  ax.scatter(*pts[0], color="black", s=26, marker="o", depthshade=False, zorder=7)
  ax.scatter(*pts[-1], color="black", s=32, marker="X", depthshade=False, zorder=7)


def panel(ax, GX, GY, Z, X_raw, m, order, positions, title, points=True,
          ticks=None, counts=None, tick_mode="nominal", alpha=0.85):
  # matplotlib does not depth-sort a line against a surface, so a route lying at
  # the terrain's own height is simply painted over by it. The arc grid passes a
  # lower alpha: the landscape reads as landscape and the route shows through.
  ax.plot_surface(GX, GY, Z, cmap=CMAP, vmin=0.0, vmax=1.0, linewidth=0,
                  antialiased=True, alpha=alpha, rstride=2, cstride=2)
  if points:
    # The values actually fitted, so the surface can be judged against them.
    ax.scatter(X_raw[:, 0], X_raw[:, 1], m, c="#33322e", s=2, alpha=0.30,
               depthshade=False)
  ax.set_zlim(0.0, 1.0)
  # The z axis is ticked in emotions rather than numbers -- the height is meant
  # to be read as a mood. Ticks come from where each emotion's paragraphs
  # actually land, so the labels stay honest under every blend.
  tk, lab, _ = zticks(order, positions, ticks, counts, tick_mode)
  ax.set_zticks(tk)
  ax.set_zticklabels(lab, fontsize=7)
  ax.set_xticklabels([]); ax.set_yticklabels([])
  ax.set_title(title, fontsize=9, pad=0)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--book", default="alice_wonderland")
  ap.add_argument("--model", default="bge-m3")
  ap.add_argument("--order", nargs="+", default=None,
                  help="Emotion order, bottom (negative) to top (positive).")
  ap.add_argument("--norm", default="rank", choices=["rank", "zscore", "minmax"],
                  help="Per-emotion normalization before blending.")
  ap.add_argument("--temp", default=0.3, type=float)
  ap.add_argument("--blend", default="softmax",
                  choices=["softmax", "project", "dominant", "pc1", "banded"],
                  help="How six emotions become one height. softmax is a "
                       "weighted MEAN and so is pulled to the middle; the other "
                       "three are not. See paragraph_mood().")
  ap.add_argument("--plane", default="pca", choices=["pca", "chain_umap"],
                  help="Which 2-D layout the surface is fitted over.")
  ap.add_argument("--beta", default=2.0, type=float,
                  help="chain_umap only: the weight on reading-order edges.")
  ap.add_argument("--interactive", action="store_true",
                  help="Write a rotatable HTML of the single mood surface with "
                       "every paragraph hoverable: reading position, mood, "
                       "dominant emotion, all six scores and its summary.")
  ap.add_argument("--window-sizes", nargs="*", type=int, default=None,
                  help="Restrict --arc-grid to these window sizes (e.g. 5). "
                       "Default: every setting saved under windows/.")
  ap.add_argument("--height", default="surface", choices=["surface", "own"],
                  help="Where a sampled paragraph's height comes from: the "
                       "fitted mood surface at its location (the arc lies on "
                       "the terrain), or the paragraph's own mood (it does not).")
  ap.add_argument("--angle-grid", action="store_true",
                  help="For each --strategies entry, one 1x4 figure of the same "
                       "arc from four cameras.")
  ap.add_argument("--strategies", nargs="+",
                  default=["chapter medoid", "douglas-peucker"],
                  help="Which sampling strategies --angle-grid renders.")
  ap.add_argument("--plane-png", action="store_true",
                  help="The 2-D layout alone -- the plane the surface is "
                       "fitted over -- coloured by reading order and by mood.")
  ap.add_argument("--chapter-sweep", action="store_true",
                  help="Grid over how coarse the narrative unit is: one medoid "
                       "paragraph per unit, units aligned to chapter breaks.")
  ap.add_argument("--units", nargs="+", type=int, default=[6, 12, 24, 48, 96],
                  help="Target unit counts for --chapter-sweep.")
  ap.add_argument("--paragraph-arc", action="store_true",
                  help="Grid of paragraph-sampling strategies: the arc runs "
                       "through real paragraphs, not pooled windows.")
  ap.add_argument("--n-points", default=60, type=int,
                  help="How many paragraphs each strategy may keep.")
  ap.add_argument("--arc-grid", action="store_true",
                  help="One panel per window setting saved under output/<book>/"
                       "<model>/windows, all on the same mood surface, so the "
                       "windowing is the only thing that varies.")
  ap.add_argument("--window-point", default="medoid", choices=["mean", "medoid"],
                  help="What stands for a window on the arc: the centroid of "
                       "its paragraphs, or the most central real paragraph.")
  ap.add_argument("--legs", default="geodesic", choices=["straight", "geodesic"],
                  help="How consecutive points are joined: a geodesic along the "
                       "surface, or a straight chord that tunnels through it.")
  ap.add_argument("--alpha", default=1.0, type=float,
                  help="Vertical exaggeration used to mesh for geodesic legs.")
  ap.add_argument("--snap-tol", default=1.0, type=float)
  ap.add_argument("--oversample", default=4.0, type=float)
  ap.add_argument("--surface-alpha", default=0.45, type=float,
                  help="Terrain opacity in --arc-grid. Low enough that the "
                       "route is not painted over by the surface it lies on.")
  ap.add_argument("--draw-lift", default=0.008, type=float,
                  help="Cosmetic: how far above the terrain the route is drawn, "
                       "so it does not z-fight with the surface it lies on.")
  ap.add_argument("--route-width", default=1.4, type=float,
                  help="Line width of the arc. Thin enough that the terrain "
                       "under it stays visible.")
  ap.add_argument("--no-points", action="store_true",
                  help="Draw the terrain alone, without the paragraphs it was "
                       "fitted to. The landscape as a landscape.")
  ap.add_argument("--ticks", default="nominal", choices=["nominal", "measured"],
                  help="nominal: the spectrum, evenly spaced -- what the height "
                       "means. measured: where each emotion's paragraphs "
                       "actually land, colliding labels dropped.")
  ap.add_argument("--compare-blends", action="store_true",
                  help="One row per blend at a fixed temperature, so the four "
                       "ways of collapsing six emotions can be judged together.")
  ap.add_argument("--hx", default=0.15, type=float)
  ap.add_argument("--hy", default=None, type=float,
                  help="Defaults to --hx (isotropic).")
  ap.add_argument("--sweep", action="store_true",
                  help="Render a temperature x bandwidth grid instead of one "
                       "surface, so the parameters can be judged together.")
  ap.add_argument("--temps", nargs="+", type=float,
                  default=[0.15, 0.3, 0.6, 1.0])
  ap.add_argument("--bandwidths", nargs="+", type=float,
                  default=[0.10, 0.15, 0.20, 0.30])
  ap.add_argument("--resolution", default=120, type=int)
  ap.add_argument("--margin", default=0.05, type=float)
  # Negative = no mask, one unbroken surface over the whole PCA rectangle. The
  # default here differs from the rest of the repo on purpose: this family is for
  # reading the mood landscape as a landscape.
  ap.add_argument("--density-floor", default=-1.0, type=float,
                  help="Percentile of paragraph support below which the surface "
                       "is masked. Negative masks nothing.")
  args = ap.parse_args()
  hy = args.hy if args.hy is not None else args.hx

  paragraphs = load_paragraphs(args.book)
  emotions, matrix = load_scores(args.book, paragraphs)
  X_raw, proj = load_plane(args.book, args.model, args.plane, args.beta)
  args.n_paragraphs = len(X_raw)
  if len(paragraphs) != len(X_raw):
    raise ValueError(f"pca ({len(X_raw)}) != paragraphs ({len(paragraphs)})")

  order = args.order or [e for e in DEFAULT_ORDER if e in emotions]
  order += [e for e in emotions if e not in order]          # never drop one

  print(f"=== mood surface, fitted to the paragraphs: {args.book} / {args.model} ===")
  print("  spectrum (bottom->top): " +
        " -> ".join(f"{e}({p:.2f})" for e, p in
                    zip(order, np.linspace(0, 1, len(order)))))

  variant = {"norm": args.norm}
  if args.plane != "pca":
    variant["plane"] = f"umap-b{args.beta:g}"
  if not args.sweep:
    variant.update({"t": args.temp, "h": args.hx,
                    "hy": hy if hy != args.hx else None})
  variant["df"] = None if args.density_floor == 5.0 else args.density_floor
  out_dir = paths.out_dir(
    args.book, args.model,
    os.path.join(paths.NARRATIVE_ARC_3D, paths.ARC_MOOD_AXIS,
                 paths.MOOD_ON_POINTS), variant)

  if args.interactive:
    m, positions, v = paragraph_mood(matrix, emotions, order, args.norm,
                                     args.temp, args.blend)
    GX, GY, Z, xs, ys = build_surface(X_raw, m, args.hx, hy, args.resolution,
                                      args.margin, args.density_floor)
    with open(os.path.join(paths.ROOT, "books", args.book,
                           "paragraph_scores.json")) as f:
      scored = json.load(f)
    summaries = []
    for i in range(len(X_raw)):
      t = (scored[i].get("summary") if i < len(scored) else None) or ""
      t = " ".join(str(t).split())
      summaries.append(t if len(t) <= 200 else t[:200] + " ...")
    chapters = None
    try:
      with open(os.path.join(paths.ROOT, "books", args.book,
                             "processed.json")) as f:
        ch = json.load(f).get("chapters", [])
      chapters = np.zeros(len(X_raw), dtype=int)
      for c in ch:
        a, b = c["paragraph_range"]
        chapters[a:b] = c.get("chapter_id", 0) + 1
    except (OSError, KeyError):
      pass
    print(f"  mood {m.min():.2f}..{m.max():.2f}  sd {m.std():.3f}")
    interactive(GX, GY, Z, X_raw, m, v, order, positions, chapters, summaries,
                emotions, matrix, args, out_dir, proj)
    paths.stamp(out_dir, __file__, args, stack="geometry.smoothers",
                order=order, renderer="plotly")
    print(f"  -> {out_dir}")
    return

  if args.angle_grid:
    # One strategy per file, four cameras across. A 3-D arc read from a single
    # fixed angle hides its own crossings: two legs that pass at different
    # heights look like an intersection, and a climb looks like a sideways move.
    # Rotating is the only way to tell those apart on paper.
    VIEWS = [(26, -60), (26, 20), (55, -45), (10, -100)]
    m, positions, v = paragraph_mood(matrix, emotions, order, args.norm,
                                     args.temp, args.blend)
    GX, GY, Z, xs, ys = build_surface(X_raw, m, args.hx, hy, args.resolution,
                                      args.margin, args.density_floor)
    ticks, counts = spectrum_ticks(m, v, positions)
    cell = float(abs(GX[0, 1] - GX[0, 0]))
    chapters = np.zeros(len(X_raw), dtype=int)
    try:
      with open(os.path.join(paths.ROOT, "books", args.book,
                             "processed.json")) as f:
        for c in json.load(f).get("chapters", []):
          a, b = c["paragraph_range"]
          chapters[a:b] = c.get("chapter_id", 0)
    except (OSError, KeyError):
      pass
    rng = np.random.default_rng(0)

    for strat in args.strategies:
      idx = sample_paragraphs(strat, X_raw, m, v, chapters, args.n_points, rng)
      P = X_raw[idx]
      zz = (m[idx] if args.height == "own"
            else surface_at(xs, ys, X_raw, m, P, args.hx, hy))
      fig = plt.figure(figsize=(5.6 * len(VIEWS), 5.6))
      for k, (elev, azim) in enumerate(VIEWS):
        ax = fig.add_subplot(1, len(VIEWS), k + 1, projection="3d")
        panel(ax, GX, GY, Z, X_raw, m, order, positions,
              f"elev {elev}\u00b0  azim {azim}\u00b0",
              points=False, ticks=ticks, counts=counts, tick_mode=args.ticks,
              alpha=args.surface_alpha)
        if args.legs == "geodesic":
          length, note = draw_route(ax, GX, GY, Z, P, idx.astype(float),
                                    args, cell)
        else:
          draw_arc(ax, P, zz, idx.astype(float), args.draw_lift,
                   args.route_width, (0, args.n_paragraphs - 1))
          note = "straight legs"
          length = float(np.linalg.norm(
            np.diff(np.column_stack([P, zz]), axis=0), axis=1).sum())
        ax.scatter(P[:, 0], P[:, 1], zz + args.draw_lift, s=8, c="black",
                   depthshade=False, zorder=8)
        ax.view_init(elev=elev, azim=azim)
      reading_bar(fig, args.n_paragraphs)
      fig.suptitle(
        f"{strat}: the same arc from four angles -- {args.book} / {args.model}\n"
        f"{len(idx)} paragraphs, {note}, on the mood surface "
        f"(blend {args.blend}, norm {args.norm}, h {args.hx:g}, "
        f"alpha {args.alpha:g})", fontsize=12)
      fig.subplots_adjust(left=0.01, right=0.99, top=0.84, bottom=0.10,
                          wspace=0.04)
      slug = strat.replace(" ", "_").replace("-", "_")
      p_out = os.path.join(out_dir, paths.named("mood_arc", f"angles_{slug}",
                                                projection=proj))
      fig.savefig(p_out, dpi=170); plt.close(fig)
      print(f"  {strat:<16} {len(idx):3d} paragraphs   {note}   "
            f"length {length:.1f}\n    wrote {p_out}")
    paths.stamp(out_dir, __file__, args, order=order,
                note="one strategy per file, four cameras")
    print(f"  -> {out_dir}")
    return

  if args.plane_png:
    # The plane on its own, before any height is put on it: the layout every
    # surface in this family is fitted over. Two colourings of the same points,
    # because the two questions the plane can answer are "when" and "what".
    m, positions, v = paragraph_mood(matrix, emotions, order, args.norm,
                                     args.temp, args.blend)
    win = np.argmax(v, axis=0)
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    lab = ("PC1", "PC2") if proj == "pca" else ("UMAP-1", "UMAP-2")

    sc = axes[0].scatter(X_raw[:, 0], X_raw[:, 1], c=np.arange(len(X_raw)),
                         cmap="plasma", s=12, linewidths=0)
    cb = fig.colorbar(sc, ax=axes[0], fraction=0.046, pad=0.02)
    cb.set_label("reading position (paragraph)")
    axes[0].set_title("by reading order", fontsize=11)

    sc2 = axes[1].scatter(X_raw[:, 0], X_raw[:, 1], c=m, cmap=CMAP,
                          vmin=0, vmax=1, s=12, linewidths=0)
    cb2 = fig.colorbar(sc2, ax=axes[1], fraction=0.046, pad=0.02,
                       ticks=list(positions))
    cb2.ax.set_yticklabels(list(order))
    cb2.set_label("mood")
    axes[1].set_title("by mood", fontsize=11)

    for ax in axes:
      ax.set_xlabel(lab[0]); ax.set_ylabel(lab[1])
      ax.set_aspect("equal", adjustable="datalim")
    fig.suptitle(f"The plane the surface is fitted over -- {args.book} / "
                 f"{args.model}\n{proj}  ({len(X_raw)} paragraphs, "
                 f"blend {args.blend}, norm {args.norm})", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    p_out = os.path.join(out_dir, paths.named("plane", "scatter", projection=proj))
    fig.savefig(p_out, dpi=170); plt.close(fig)
    print(f"  wrote {p_out}")
    paths.stamp(out_dir, __file__, args, order=order, note="the 2-D plane alone")
    print(f"  -> {out_dir}")
    return

  if args.chapter_sweep:
    # How coarse does the narrative unit have to be before the arc is legible?
    # Range was never the problem once we left windows behind; continuity is.
    m, positions, v = paragraph_mood(matrix, emotions, order, args.norm,
                                     args.temp, args.blend)
    GX, GY, Z, xs, ys = build_surface(X_raw, m, args.hx, hy, args.resolution,
                                      args.margin, args.density_floor)
    ticks, counts = spectrum_ticks(m, v, positions)
    cell = float(abs(GX[0, 1] - GX[0, 0]))
    chapters = np.zeros(len(X_raw), dtype=int)
    with open(os.path.join(paths.ROOT, "books", args.book, "processed.json")) as f:
      ch = json.load(f).get("chapters", [])
    for c in ch:
      a, b = c["paragraph_range"]
      chapters[a:b] = c.get("chapter_id", 0)
    print(f"  {len(ch)} chapters, {len(X_raw)} paragraphs")

    cols = min(3, len(args.units))
    rows = int(np.ceil(len(args.units) / cols))
    fig = plt.figure(figsize=(6.0 * cols, 5.4 * rows))
    summary = []
    for k, target in enumerate(args.units):
      units = narrative_units(chapters, target)
      idx = unit_medoids(X_raw, units)
      P = X_raw[idx]
      # On the surface, not at the paragraph's own mood: the arc is supposed to
      # lie on the terrain, and a medoid's own score is one paragraph's opinion
      # rather than the mood of the region it stands in.
      zz = (m[idx] if args.height == "own"
            else surface_at(xs, ys, X_raw, m, P, args.hx, hy))
      ax = fig.add_subplot(rows, cols, k + 1, projection="3d")
      sizes = [len(u) for u in units]
      panel(ax, GX, GY, Z, X_raw, m, order, positions,
            f"{len(idx)} units   ~{int(np.median(sizes))} paragraphs each",
            points=False, ticks=ticks, counts=counts, tick_mode=args.ticks,
            alpha=args.surface_alpha)
      if args.legs == "geodesic":
        length, note = draw_route(ax, GX, GY, Z, P, idx.astype(float), args, cell)
      else:
        draw_arc(ax, P, zz, idx.astype(float), args.draw_lift,
                 args.route_width, (0, args.n_paragraphs - 1))
        note = "straight legs"
        length = float(np.linalg.norm(np.diff(np.column_stack([P, zz]), axis=0),
                                      axis=1).sum())
      ax.scatter(P[:, 0], P[:, 1], zz + args.draw_lift, s=10, c="black",
                 depthshade=False, zorder=8)

      dz = float(np.abs(np.diff(zz)).mean())
      summary.append({"units": int(len(idx)),
                      "median_unit_paragraphs": int(np.median(sizes)),
                      "mood_min": float(zz.min()), "mood_max": float(zz.max()),
                      "mood_span": float(zz.max() - zz.min()),
                      "mean_abs_mood_step": dz, "route_length": length})
      print(f"  {len(idx):3d} units (~{int(np.median(sizes)):3d} paragraphs)   "
            f"mood {zz.min():.2f}..{zz.max():.2f}   "
            f"mean |mood step| {dz:.3f}   {note}   length {length:.1f}")

    fig.suptitle(
      f"The narrative arc through chapter units -- {args.book} / {args.model}\n"
      f"one medoid paragraph per unit, units respect chapter breaks "
      f"(blend {args.blend}, norm {args.norm}, h {args.hx:g})\n"
      f"route coloured by reading position; z = mood, sadness to humor",
      fontsize=12)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.03,
                        wspace=0.06, hspace=0.12)
    reading_bar(fig, args.n_paragraphs)
    p_out = os.path.join(out_dir, paths.named("mood_arc", "chapter_sweep",
                                              projection=proj))
    fig.savefig(p_out, dpi=170); plt.close(fig)
    print(f"\n  wrote {p_out}")
    with open(os.path.join(out_dir, "chapter_sweep_metrics.json"), "w") as f:
      json.dump(summary, f, indent=2)
    paths.stamp(out_dir, __file__, args, order=order,
                note="arc through one medoid per narrative unit")
    print(f"  -> {out_dir}")
    return

  if args.paragraph_arc:
    # The arc through *paragraphs*, not windows: no pooling anywhere, so each
    # point is one real annotation at one real position. The only question left
    # is which paragraphs to draw it through, which is what the panels compare.
    STRATEGIES = ["uniform", "chapter medoid", "mood extrema", "douglas-peucker",
                  "decisive", "farthest point"]
    m, positions, v = paragraph_mood(matrix, emotions, order, args.norm,
                                     args.temp, args.blend)
    GX, GY, Z, xs, ys = build_surface(X_raw, m, args.hx, hy, args.resolution,
                                      args.margin, args.density_floor)
    ticks, counts = spectrum_ticks(m, v, positions)
    cell = float(abs(GX[0, 1] - GX[0, 0]))
    chapters = np.zeros(len(X_raw), dtype=int)
    try:
      with open(os.path.join(paths.ROOT, "books", args.book,
                             "processed.json")) as f:
        for c in json.load(f).get("chapters", []):
          a, b = c["paragraph_range"]
          chapters[a:b] = c.get("chapter_id", 0)
    except (OSError, KeyError):
      pass
    rng = np.random.default_rng(0)

    cols = min(3, len(STRATEGIES))
    rows = int(np.ceil(len(STRATEGIES) / cols))
    fig = plt.figure(figsize=(6.0 * cols, 5.4 * rows))
    summary = []
    for k, strat in enumerate(STRATEGIES):
      idx = sample_paragraphs(strat, X_raw, m, v, chapters, args.n_points, rng)
      P = X_raw[idx]
      zz = (m[idx] if args.height == "own"
            else surface_at(xs, ys, X_raw, m, P, args.hx, hy))
      ax = fig.add_subplot(rows, cols, k + 1, projection="3d")
      panel(ax, GX, GY, Z, X_raw, m, order, positions,
            f"{strat}   {len(idx)} paragraphs",
            points=False, ticks=ticks, counts=counts, tick_mode=args.ticks,
            alpha=args.surface_alpha)
      if args.legs == "geodesic":
        length_g, note = draw_route(ax, GX, GY, Z, P, idx.astype(float), args, cell)
      else:
        draw_arc(ax, P, zz, idx.astype(float), args.draw_lift,
                 args.route_width, (0, args.n_paragraphs - 1))
        length_g, note = None, "straight legs"
      ax.scatter(P[:, 0], P[:, 1], zz + args.draw_lift, s=6, c="black",
                 depthshade=False, zorder=8)

      span = float(zz.max() - zz.min())
      # How much of the terrain the sample walks over, as the area of its
      # bounding box against the whole cloud's.
      cover = (float(np.ptp(P[:, 0]) * np.ptp(P[:, 1])) /
               float(np.ptp(X_raw[:, 0]) * np.ptp(X_raw[:, 1])))
      length = length_g if length_g is not None else float(
        np.linalg.norm(np.diff(np.column_stack([P, zz]), axis=0), axis=1).sum())
      summary.append({"strategy": strat, "legs": args.legs, "n": int(len(idx)),
                      "mood_min": float(zz.min()), "mood_max": float(zz.max()),
                      "mood_span": span, "plane_coverage": cover,
                      "route_length": length})
      print(f"  {strat:<16} {len(idx):4d} paragraphs   mood "
            f"{zz.min():.2f}..{zz.max():.2f} (span {span:.2f})   "
            f"covers {100*cover:4.0f}% of the plane   {note}   "
            f"length {length:.1f}")

    fig.suptitle(
      f"The narrative arc through paragraphs -- {args.book} / {args.model}\n"
      f"same terrain and same reading order in every panel "
      f"(blend {args.blend}, norm {args.norm}, h {args.hx:g}); "
      f"only which paragraphs are sampled changes\n"
      f"route coloured by reading position; z = mood, sadness to humor",
      fontsize=12)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.03,
                        wspace=0.06, hspace=0.12)
    reading_bar(fig, args.n_paragraphs)
    p_out = os.path.join(out_dir, paths.named("mood_arc", "by_sampling",
                                              projection=proj))
    fig.savefig(p_out, dpi=170); plt.close(fig)
    print(f"\n  wrote {p_out}")
    with open(os.path.join(out_dir, "sampling_metrics.json"), "w") as f:
      json.dump(summary, f, indent=2)
    paths.stamp(out_dir, __file__, args, order=order,
                note="arc through sampled paragraphs, one panel per strategy")
    print(f"  -> {out_dir}")
    return

  if args.arc_grid:
    # One terrain, many windowings. The mood surface does not depend on the
    # window at all -- it is fitted to paragraphs -- so every panel shows the
    # *same* landscape and only the route over it changes. That is the point of
    # the grid: it isolates what the windowing does, with the surface held fixed.
    settings = window_settings(args.book, args.model)
    if args.window_sizes:
      settings = [s for s in settings if s["size"] in args.window_sizes]
    if not settings:
      raise SystemExit(f"no saved window settings under output/{args.book}/"
                       f"{args.model}/{paths.WINDOWS} -- run windows.py first")
    m, positions, v = paragraph_mood(matrix, emotions, order, args.norm,
                                     args.temp, args.blend)
    GX, GY, Z, xs, ys = build_surface(X_raw, m, args.hx, hy, args.resolution,
                                      args.margin, args.density_floor)
    ticks, counts = spectrum_ticks(m, v, positions)
    cell = float(abs(GX[0, 1] - GX[0, 0]))

    # One figure per window SIZE: the sizes are not comparable panel-to-panel
    # (w5 and w100 differ by 20x in how much text each point averages), while
    # the strides within a size are exactly the comparison worth making side by
    # side. So each size gets its own grid rather than one mixed sheet.
    sizes = sorted({st["size"] for st in settings})
    summary = []
    for size in sizes:
      group = [st for st in settings if st["size"] == size]
      group.sort(key=lambda r: r["stride"] or 0)
      cols = min(3, len(group))
      rows = int(np.ceil(len(group) / cols))
      fig = plt.figure(figsize=(6.0 * cols, 5.4 * rows))
      for k, st in enumerate(group):
        wins = st["windows"]
        arc_xy, arc_idx = window_points(X_raw, wins, args.window_point)
        centers = np.array([(a + b - 1) / 2.0 for a, b in wins])
        ax = fig.add_subplot(rows, cols, k + 1, projection="3d")
        panel(ax, GX, GY, Z, X_raw, m, order, positions,
              f"stride {st['stride']}   {len(wins)} windows",
              points=False, ticks=ticks, counts=counts, tick_mode=args.ticks,
              alpha=args.surface_alpha)

        if args.legs == "geodesic":
          runs, gst = geodesic_route(GX, GY, Z, arc_xy, args.alpha,
                                     args.snap_tol * cell)
          for P, tt in runs:
            draw_arc(ax, P[:, :2], P[:, 2],
                     np.interp(tt, np.arange(len(centers)), centers),
                     args.draw_lift)
          note = f"{gst['legs_solved']}/{gst['legs_total']} geodesic legs"
          length = sum(float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())
                       for P, _ in runs)
        else:
          dense_xy, t = resample(arc_xy, cell / args.oversample)
          z = surface_at(xs, ys, X_raw, m, dense_xy, args.hx, hy)
          draw_arc(ax, dense_xy, z,
                   np.interp(t, np.arange(len(centers)), centers),
                   args.draw_lift, args.route_width, (0, args.n_paragraphs - 1))
          note = "straight legs, lifted"
          P = np.column_stack([dense_xy, z])
          length = float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())

        z_win = surface_at(xs, ys, X_raw, m, arc_xy, args.hx, hy)
        summary.append({"windows": st["name"], "size": size,
                        "stride": st["stride"], "n_windows": len(wins),
                        "legs": args.legs, "route_length": length,
                        "mood_at_windows_min": float(z_win.min()),
                        "mood_at_windows_max": float(z_win.max()),
                        "mood_at_windows_sd": float(z_win.std())})
        print(f"  {st['name']:<10} {len(wins):4d} windows  {note}   "
              f"mood {z_win.min():.2f}..{z_win.max():.2f} "
              f"(sd {z_win.std():.3f})   length {length:.2f}")

      fig.suptitle(
        f"The narrative arc on one mood surface -- window {size} -- "
        f"{args.book} / {args.model}\n"
        f"same terrain in every panel (blend {args.blend}, norm {args.norm}, "
        f"h {args.hx:g}); only the stride changes\n"
        f"route coloured by reading position; z = mood, sadness (low) to "
        f"humor (high)", fontsize=12)
      fig.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.03,
                          wspace=0.06, hspace=0.12)
      reading_bar(fig, args.n_paragraphs)
      p_out = os.path.join(out_dir, paths.named("mood_arc", f"grid_w{size}", projection=proj))
      fig.savefig(p_out, dpi=170); plt.close(fig)
      print(f"  wrote {p_out}\n")

    with open(os.path.join(out_dir, "arc_grid_metrics.json"), "w") as f:
      json.dump(summary, f, indent=2)
    paths.stamp(out_dir, __file__, args, stack="geometry.smoothers",
                order=order, note="one arc grid per window size")
    print(f"  -> {out_dir}")
    return

  if args.compare_blends:
    blends = ["softmax", "project", "pc1", "dominant", "banded"]
    combos = [(b, args.temp, h) for b in blends for h in bws_default(args)]
    rows, cols = len(blends), len(bws_default(args))
  else:
    temps = args.temps if args.sweep else [args.temp]
    bws = args.bandwidths if args.sweep else [args.hx]
    combos = [(args.blend, t, h) for t in temps for h in bws]
    rows, cols = len(temps), len(bws)

  # A single panel gets a proper canvas: at grid proportions the title collides
  # with the axes and the surface is postage-stamp sized.
  single = rows * cols == 1
  fig = plt.figure(figsize=(10, 8.5) if single else (4.6 * cols, 4.4 * rows))
  summary = []
  for k, (b, t, h) in enumerate(combos):
    m, positions, v = paragraph_mood(matrix, emotions, order, args.norm, t, b)
    hyy = hy if not (args.sweep or args.compare_blends) else h
    GX, GY, Z, xs, ys = build_surface(X_raw, m, h, hyy, args.resolution,
                                      args.margin, args.density_floor)
    ticks, counts = spectrum_ticks(m, v, positions)
    ax = fig.add_subplot(rows, cols, k + 1, projection="3d")
    label = ("" if rows * cols == 1 else
             (f"{b}   h {h:g}" if args.compare_blends else f"temp {t:g}   h {h:g}"))
    panel(ax, GX, GY, Z, X_raw, m, order, positions, label,
          points=not args.no_points, ticks=ticks, counts=counts,
          tick_mode=args.ticks)
    if args.ticks == "measured":
      _, _, dropped = zticks(order, positions, ticks, counts, "measured")
      if dropped:
        print(f"           labels not drawn (they collide): {', '.join(dropped)}")
    # How much of the axis the emotions actually occupy: the gap between the
    # lowest and highest emotion's typical height. Small = unreadable, whatever
    # the surface looks like.
    sep = float(ticks.max() - ticks.min()) if len(ticks) else 0.0
    rec = {"blend": b, "temp": float(t), "h": float(h), "norm": args.norm,
           "mood_min": float(m.min()), "mood_max": float(m.max()),
           "mood_sd": float(m.std()), "emotion_separation": sep,
           "surface_min": float(np.nanmin(Z)), "surface_max": float(np.nanmax(Z)),
           "surface_sd": float(np.nanstd(Z))}
    summary.append(rec)
    print(f"  {b:<8} temp {t:<5g} h {h:<5g}  mood sd {m.std():.3f}   "
          f"surface sd {np.nanstd(Z):.3f}   emotions span {sep:.2f} of the axis")

  head = f"The mood surface, fitted to the paragraphs -- {args.book} / {args.model}"
  if single:
    b, t, h = combos[0]
    title = (f"{head}\nblend {b}, norm {args.norm}"
             + (f", temp {t:g}" if b == "softmax" else "")
             + f", h {h:g}\nz = mood: 0.0 = sadness ... 1.0 = humor, "
             + f"the emotions evenly spaced between")
  else:
    title = (f"{head}\none value per paragraph (norm {args.norm}), then one "
             f"Gaussian NW surface through them;\n"
             f"z ticks sit where each emotion's paragraphs actually land")
  fig.suptitle(title, fontsize=12)
  fig.subplots_adjust(left=0.02, right=0.92 if single else 0.98,
                      top=0.90 if single else (0.90 if rows > 1 else 0.84),
                      bottom=0.03, wspace=0.06, hspace=0.12)
  name = paths.named("mood_surface", "sweep" if args.sweep else "single",
                     projection=proj)
  p = os.path.join(out_dir, name)
  fig.savefig(p, dpi=170); plt.close(fig)
  print(f"\n  wrote {p}")

  with open(os.path.join(out_dir, "metrics.json"), "w") as f:
    json.dump(summary, f, indent=2)
  paths.stamp(out_dir, __file__, args, stack="geometry.smoothers",
              estimator="gaussian_nw (isotropic h unless --hy; standardized)",
              order=order, blend="softmax over normalized emotions, then fit",
              note="mood collapsed per paragraph BEFORE smoothing")
  print(f"  -> {out_dir}")


if __name__ == "__main__":
  main()
