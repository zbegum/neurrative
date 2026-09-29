"""
Is the narrative arc a real feature of the embedding, or an artifact of the
timeseries window?

The narrative arc (windows.py; drawn by narrative-arc/) slides a window over
paragraphs in reading order, mean-pooling each window, and connecting the pooled
points. The pooling is a low-pass filter: it smooths paragraph-to-paragraph
jitter into a clean curve. The worry is circular -- if you smooth hard enough,
*any* sequence looks like an arc. So the question this script asks is:

    does the same arc survive WITHOUT the window? i.e. is it already present in
    the raw, per-paragraph embedding (window = 1, no pooling)?

The comparison is made honest by one fact: PCA is linear, so projecting a pooled
window equals averaging the projected paragraphs. The windowed arc therefore
lives in the *same* PCA coordinates as the per-paragraph trajectory -- it is
literally a moving average of it. That lets us overlay the two and measure how
much of each paragraph's position is the arc (the slow trend) versus local
jitter (the fast residual). A high fraction means the arc is a genuine feature
of the non-timeseries embedding; the window only made it easier to see.

Three pieces of evidence, none of which depend on the window to define the arc:

  variance explained   in the full 1024-d embedding, what fraction of a
                       paragraph's deviation from the mean is captured by the
                       smoothed (arc) component. Projection-free.
  path wiggliness      how much longer the raw per-paragraph path is than the
                       arc -- the jitter the window removed.
  temporal coherence   cosine similarity of paragraphs d apart vs. lag d, next
                       to a reading-order-shuffled baseline. Slow decay above
                       the shuffle line = a real trajectory, not smoothing.

Everything lands in output/<book>/<model>/arc_comparison/.

Example:

  python visualization/arc_comparison.py --book alice_wonderland --model bge-m3
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
from sklearn.decomposition import PCA

import paths
from data import load_paragraphs
from windows import window_bounds, window_centers


def smoothed_trajectory(rows, size):
  """Centered moving average of width `size` at every paragraph.

  This is the arc signal at full paragraph resolution: the windowed/strided arc
  is just a subsample of it. Edges shrink the window rather than pad, so the
  opening and closing paragraphs are averaged over what actually exists.
  """
  n = len(rows)
  half = size // 2
  out = np.empty_like(rows, dtype=np.float64)
  for i in range(n):
    lo, hi = max(0, i - half), min(n, i + half + 1)
    out[i] = rows[lo:hi].mean(axis=0)
  return out


def variance_explained(raw, smooth):
  """Fraction of per-paragraph variance (summed over dims) the arc captures.

  1 - Var(raw - smooth) / Var(raw), both measured as total variance about the
  mean across paragraphs. Projection-free: computed in the native embedding
  dimension. High => the arc is where the paragraphs actually are.
  """
  total = np.var(raw - raw.mean(axis=0), axis=0).sum()
  resid = np.var(raw - smooth, axis=0).sum()
  return float(1.0 - resid / total)


def path_length(xy):
  return float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())


def step_autocorr(xy):
  """Mean cosine between consecutive unit steps -- how smoothly the path turns."""
  steps = np.diff(xy, axis=0)
  norms = np.linalg.norm(steps, axis=1, keepdims=True)
  u = steps / np.clip(norms, 1e-12, None)
  return float(np.mean(np.sum(u[:-1] * u[1:], axis=1)))


def temporal_coherence(rows, max_lag, seed):
  """Mean cosine similarity of paragraphs `lag` apart, for lag = 1..max_lag.

  Returned next to a baseline where the reading order is shuffled: if the arc
  were an artifact of smoothing rather than a real trajectory, the real curve
  would sit on top of the shuffled one. L2-normalize first so this is cosine.
  """
  r = rows / np.clip(np.linalg.norm(rows, axis=1, keepdims=True), 1e-12, None)
  lags = np.arange(1, max_lag + 1)

  def curve(mat):
    return np.array([np.mean(np.sum(mat[:-l] * mat[l:], axis=1)) for l in lags])

  real = curve(r)
  rng = np.random.default_rng(seed)
  shuf = np.mean([curve(r[rng.permutation(len(r))]) for _ in range(20)], axis=0)
  return lags, real, shuf


def draw_arc(ax, xy, title, lw=2.4, points=True):
  """A reading-order path colored by progression, with O (open) and X (end)."""
  seg = np.stack([xy[:-1], xy[1:]], axis=1)
  lc = LineCollection(seg, cmap="plasma", linewidth=lw, alpha=0.9, zorder=2)
  lc.set_array(np.linspace(0, 1, len(xy) - 1))
  ax.add_collection(lc)
  if points:
    ax.scatter(xy[:, 0], xy[:, 1], c=np.linspace(0, 1, len(xy)), cmap="plasma",
               s=14, alpha=0.7, linewidths=0.2, edgecolors="white", zorder=3)
  ax.scatter(*xy[0], s=130, facecolor="none", edgecolor="black",
             linewidths=1.6, marker="o", zorder=4)
  ax.annotate("O", xy[0], fontsize=9, fontweight="bold", ha="center",
              va="center", zorder=5)
  ax.scatter(*xy[-1], s=150, color="black", marker="X", zorder=4)
  ax.set_title(title, fontsize=11)
  ax.set_xlabel("PC1")
  ax.set_ylabel("PC2")


def same_limits(axes, xy_list, pad=0.05):
  """Put every panel on one scale so the two arcs are visually comparable."""
  allxy = np.vstack(xy_list)
  lo, hi = allxy.min(axis=0), allxy.max(axis=0)
  span = (hi - lo) * pad
  lo, hi = lo - span, hi + span
  for ax in axes:
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_aspect("equal", adjustable="box")


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--book", default="alice_wonderland")
  ap.add_argument("--model", default="bge-m3")
  ap.add_argument("--size", default=10, type=int,
                  help="Timeseries window length in paragraphs.")
  ap.add_argument("--stride", default=5, type=int,
                  help="Paragraphs between windows (matches an existing arc).")
  ap.add_argument("--max-lag", default=60, type=int,
                  help="Largest paragraph lag for the temporal-coherence curve.")
  ap.add_argument("--seed", default=0, type=int)
  args = ap.parse_args()

  paragraphs = load_paragraphs(args.book)
  emb = np.load(os.path.join("books", args.book, "embeddings", args.model,
                             "embeddings.npy")).astype(np.float64)
  if len(emb) != len(paragraphs):
    raise ValueError(f"embeddings ({len(emb)}) != paragraphs ({len(paragraphs)})")
  n = len(emb)

  out_dir = paths.out_dir(args.book, args.model, paths.ARC_COMPARISON)
  paths.stamp(out_dir, __file__, args)
  print(f"=== arc comparison: {args.book} / {args.model} ===")
  print(f"  {n} paragraphs, {emb.shape[1]}-d embeddings, "
        f"window size {args.size} stride {args.stride}")

  # One PCA basis, fit on the raw per-paragraph embeddings. Both arcs are drawn
  # in it, so the windowed arc is a moving average of the per-paragraph one and
  # the two overlay exactly.
  pca = PCA(n_components=2).fit(emb)
  var = pca.explained_variance_ratio_
  para_xy = pca.transform(emb)                       # non-timeseries (window = 1)

  windows = window_bounds(n, args.size, args.stride)
  pooled = np.array([emb[s:e].mean(axis=0) for s, e in windows])
  arc_xy = pca.transform(pooled)                     # timeseries (windowed)

  # Sanity: projecting the pool == averaging the projection (PCA is linear).
  centers = window_centers(windows)
  avg_of_proj = np.array([para_xy[s:e].mean(axis=0) for s, e in windows])
  identity_gap = float(np.abs(arc_xy - avg_of_proj).max())
  print(f"  linearity check |pool-then-PCA - PCA-then-pool| max = {identity_gap:.2e}")

  # --- metrics -----------------------------------------------------------------
  smooth_full = smoothed_trajectory(emb, args.size)
  ve_full = variance_explained(emb, smooth_full)
  ve_pca = variance_explained(para_xy, smoothed_trajectory(para_xy, args.size))

  wiggle = path_length(para_xy) / path_length(arc_xy)
  ac_para, ac_arc = step_autocorr(para_xy), step_autocorr(arc_xy)
  lags, coh_real, coh_shuf = temporal_coherence(emb, min(args.max_lag, n - 1),
                                                args.seed)

  metrics = {
    "book": args.book, "model": args.model,
    "n_paragraphs": n, "embedding_dim": int(emb.shape[1]),
    "window_size": args.size, "window_stride": args.stride, "n_windows": len(windows),
    "pca_var_pc1": float(var[0]), "pca_var_pc2": float(var[1]),
    "variance_explained_by_arc_fulldim": ve_full,
    "variance_explained_by_arc_pca2d": ve_pca,
    "path_wiggliness_ratio": wiggle,
    "step_autocorr_perparagraph": ac_para,
    "step_autocorr_arc": ac_arc,
    "temporal_coherence_lag1_real": float(coh_real[0]),
    "temporal_coherence_lag1_shuffled": float(coh_shuf[0]),
    "linearity_check_max_gap": identity_gap,
  }
  with open(os.path.join(out_dir, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

  print(f"  arc explains {ve_full:.1%} of full-dim per-paragraph variance "
        f"({ve_pca:.1%} in the PC1-PC2 plane)")
  print(f"  per-paragraph path is {wiggle:.1f}x longer than the arc (jitter removed)")
  print(f"  step autocorrelation: per-paragraph {ac_para:+.3f}  arc {ac_arc:+.3f}")
  print(f"  temporal coherence lag 1: real {coh_real[0]:.3f} vs "
        f"shuffled {coh_shuf[0]:.3f}")

  # --- window-size sweep: the smoothing tradeoff --------------------------------
  # A bigger window draws a cleaner arc but is a heavier low-pass filter, so it
  # keeps a SMALLER fraction of where the paragraphs actually are. This makes the
  # tension explicit: legibility is bought with variance.
  sizes = [s for s in (1, 5, 10, 15, 25, 40, 60, 80) if s <= n]
  sweep = []
  for s in sizes:
    sm = smoothed_trajectory(emb, s)
    ve = variance_explained(emb, sm)
    wins = window_bounds(n, s, max(1, s // 2))
    axy = pca.transform(np.array([emb[a:b].mean(axis=0) for a, b in wins]))
    wig = path_length(para_xy) / path_length(axy) if len(axy) > 1 else float("nan")
    sweep.append({"size": s, "variance_explained_fulldim": ve, "wiggliness": wig})
    print(f"    window {s:>3}: arc keeps {ve:5.1%} of variance, "
          f"path {wig:5.1f}x smoother")
  metrics["window_sweep"] = sweep
  with open(os.path.join(out_dir, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

  fig, ax1 = plt.subplots(figsize=(8.4, 5.2))
  ss = [r["size"] for r in sweep]
  ax1.plot(ss, [r["variance_explained_fulldim"] for r in sweep],
           color="#2a78d6", lw=2, marker="o", label="variance kept by arc")
  ax1.set_xlabel("window size (paragraphs)")
  ax1.set_ylabel("fraction of per-paragraph variance kept", color="#2a78d6")
  ax1.tick_params(axis="y", labelcolor="#2a78d6")
  ax1.set_ylim(0, 1)
  ax2 = ax1.twinx()
  ax2.plot(ss, [r["wiggliness"] for r in sweep],
           color="#d1495b", lw=2, marker="s", label="path smoothed away")
  ax2.set_ylabel("per-paragraph path length / arc length", color="#d1495b")
  ax2.tick_params(axis="y", labelcolor="#d1495b")
  ax1.set_title("The smoothing tradeoff: a cleaner arc keeps less of the embedding\n"
                "bigger window -> smoother arc (red up) but less variance (blue down)",
                fontsize=11)
  fig.tight_layout()
  p = os.path.join(out_dir, "arc_window_sweep.png")
  fig.savefig(p, dpi=200); plt.close(fig); print(f"  wrote {p}")

  # --- figure 1: side by side --------------------------------------------------
  fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 7.4))
  draw_arc(a1, para_xy,
           f"Non-timeseries: per-paragraph (window = 1)\n{n} points, raw reading-order path")
  draw_arc(a2, arc_xy,
           f"Timeseries: windowed arc (size {args.size}, stride {args.stride})\n"
           f"{len(windows)} pooled windows")
  same_limits([a1, a2], [para_xy, arc_xy])
  fig.suptitle(
    f"Same PCA space ({var.sum():.0%} variance). The windowed arc is a moving "
    f"average of the per-paragraph path.\n"
    f"Arc captures {ve_full:.0%} of full-dim per-paragraph variance -- it is "
    f"already present without the window.", fontsize=12)
  fig.tight_layout(rect=(0, 0, 1, 0.94))
  p = os.path.join(out_dir, "arc_side_by_side_pca.png")
  fig.savefig(p, dpi=200); plt.close(fig); print(f"  wrote {p}")

  # --- figure 2: overlay -------------------------------------------------------
  fig, ax = plt.subplots(figsize=(8.4, 8))
  ax.plot(para_xy[:, 0], para_xy[:, 1], color="0.62", lw=0.6, alpha=0.7,
          zorder=1, label="per-paragraph path (window = 1)")
  ax.scatter(para_xy[:, 0], para_xy[:, 1], c=np.linspace(0, 1, n), cmap="plasma",
             s=9, alpha=0.4, zorder=2)
  draw_arc(ax, arc_xy, "", lw=3.0, points=False)
  ax.plot([], [], color="#d1495b", lw=3, label=f"windowed arc (size {args.size})")
  ax.set_title(f"The arc is the slow trend of the per-paragraph trajectory\n"
               f"{args.book} / {args.model}", fontsize=11)
  ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
  ax.set_aspect("equal", adjustable="box")
  ax.legend(loc="best", fontsize=9, framealpha=0.9)
  fig.tight_layout()
  p = os.path.join(out_dir, "arc_overlay_pca.png")
  fig.savefig(p, dpi=200); plt.close(fig); print(f"  wrote {p}")

  # --- figure 3: temporal coherence -------------------------------------------
  fig, ax = plt.subplots(figsize=(8.4, 5.2))
  ax.plot(lags, coh_real, color="#2a78d6", lw=2, marker="o", ms=3,
          label="reading order (real)")
  ax.plot(lags, coh_shuf, color="0.5", lw=1.6, ls="--",
          label="shuffled order (baseline)")
  ax.axhline(0, color="0.7", lw=0.6)
  ax.set_xlabel("paragraph lag d")
  ax.set_ylabel("mean cosine similarity of paragraphs d apart")
  ax.set_title("Temporal coherence in the raw embedding (no window, no PCA)\n"
               "real curve above the shuffle line = a genuine trajectory",
               fontsize=11)
  ax.legend(loc="best", fontsize=9)
  fig.tight_layout()
  p = os.path.join(out_dir, "arc_temporal_coherence.png")
  fig.savefig(p, dpi=200); plt.close(fig); print(f"  wrote {p}")

  print(f"  -> {out_dir}")


if __name__ == "__main__":
  main()
