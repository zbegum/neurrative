"""
The arc comparison, but in UMAP and t-SNE as well as PCA.

arc_comparison.py answered the question in PCA, where a clean linear identity
holds (the windowed arc is exactly a moving average of the per-paragraph path).
UMAP and t-SNE are nonlinear and refit separately on the pooled points, so that
identity is gone and two extra hazards appear:

  1. Overlap smoothness. With stride < size, neighbouring windows share
     paragraphs, so consecutive pooled points are close *by construction* -- any
     windowed path looks smooth, story or not.
  2. Manufactured structure. t-SNE (and UMAP) will draw a tidy curve out of
     noise. A pretty windowed arc is therefore weak evidence on its own.

Both hazards are met with the same control: pool over a *shuffled* paragraph
order using the identical window structure, refit the projection, and connect in
window order. The overlap smoothness survives shuffling (same sharing), and so
does whatever the projection manufactures -- so anything left over,

     coherence(real reading order) - coherence(shuffled order),

is the part the actual narrative order is responsible for. Coherence here is the
mean cosine between consecutive steps of the path (how smoothly it turns).

Outputs go next to the PCA comparison, in arc/output/<book>/<model>/arc_comparison/.

Example:

  python arc/validation/arc_comparison_projections.py --book alice_wonderland --model bge-m3
"""

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# The repository root and common/ on the path, wherever this script lives.
_ROOT = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_ROOT, "common")):
  _ROOT = os.path.dirname(_ROOT)
sys.path[1:1] = [_ROOT, os.path.join(_ROOT, "common")]

import paths
from data import load_paragraphs
from windows import pool, window_bounds


def step_autocorr(xy):
  """Mean cosine between consecutive unit steps -- higher = smoother turning."""
  steps = np.diff(xy, axis=0)
  u = steps / np.clip(np.linalg.norm(steps, axis=1, keepdims=True), 1e-12, None)
  return float(np.mean(np.sum(u[:-1] * u[1:], axis=1)))


def fit(method, X, seed):
  """2-D projection. init='pca' keeps t-SNE/UMAP runs stable across calls."""
  if method == "PCA":
    return PCA(n_components=2).fit_transform(X)
  if method == "t-SNE":
    perp = min(30, len(X) - 1)
    return TSNE(n_components=2, perplexity=perp, init="pca",
                random_state=seed).fit_transform(X)
  if method == "UMAP":
    from umap import UMAP
    return UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                random_state=seed).fit_transform(X)
  raise ValueError(method)


def draw_arc(ax, xy, title, points=True, lw=2.4):
  seg = np.stack([xy[:-1], xy[1:]], axis=1)
  lc = LineCollection(seg, cmap="plasma", linewidth=lw, alpha=0.9, zorder=2)
  lc.set_array(np.linspace(0, 1, len(xy) - 1))
  ax.add_collection(lc)
  if points:
    ax.scatter(xy[:, 0], xy[:, 1], c=np.linspace(0, 1, len(xy)), cmap="plasma",
               s=12, alpha=0.6, linewidths=0.2, edgecolors="white", zorder=3)
  ax.scatter(*xy[0], s=120, facecolor="none", edgecolor="black",
             linewidths=1.6, marker="o", zorder=4)
  ax.annotate("O", xy[0], fontsize=9, fontweight="bold", ha="center",
              va="center", zorder=5)
  ax.scatter(*xy[-1], s=140, color="black", marker="X", zorder=4)
  ax.set_title(title, fontsize=10.5)
  ax.set_xticks([]); ax.set_yticks([])
  ax.set_aspect("equal", adjustable="datalim")


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--book", default="alice_wonderland")
  ap.add_argument("--model", default="bge-m3")
  ap.add_argument("--size", default=10, type=int)
  ap.add_argument("--stride", default=5, type=int)
  ap.add_argument("--shuffles", default=5, type=int,
                  help="Shuffled-order control repeats for the coherence score.")
  ap.add_argument("--seed", default=0, type=int)
  ap.add_argument("--methods", nargs="+", default=["PCA", "UMAP", "t-SNE"])
  args = ap.parse_args()

  paragraphs = load_paragraphs(args.book)
  emb = np.load(os.path.join("books", args.book, "embeddings", args.model,
                             "embeddings.npy")).astype(np.float64)
  if len(emb) != len(paragraphs):
    raise ValueError(f"embeddings ({len(emb)}) != paragraphs ({len(paragraphs)})")
  n = len(emb)

  out_dir = paths.out_dir(args.book, args.model, paths.ARC_COMPARISON)
  paths.stamp(out_dir, __file__, args)
  windows = window_bounds(n, args.size, args.stride)
  pooled_real = pool(emb, windows)
  print(f"=== arc in projections: {args.book} / {args.model} ===")
  print(f"  {n} paragraphs -> {len(windows)} windows (size {args.size} "
        f"stride {args.stride}); {args.shuffles} shuffle controls\n")

  rng = np.random.default_rng(args.seed)
  summary = []

  for method in args.methods:
    para_xy = fit(method, emb, args.seed)               # non-timeseries
    arc_xy = fit(method, pooled_real, args.seed)        # timeseries (real order)

    ac_para = step_autocorr(para_xy)
    ac_arc = step_autocorr(arc_xy)

    # Control: same window structure over shuffled paragraph order.
    ac_shuf = []
    for _ in range(args.shuffles):
      pooled_s = pool(emb[rng.permutation(n)], windows)
      ac_shuf.append(step_autocorr(fit(method, pooled_s, args.seed)))
    ac_shuf_mean = float(np.mean(ac_shuf))
    excess = ac_arc - ac_shuf_mean          # coherence due to reading order

    summary.append({
      "method": method,
      "autocorr_perparagraph": ac_para,
      "autocorr_arc_real": ac_arc,
      "autocorr_arc_shuffled": ac_shuf_mean,
      "reading_order_excess": excess,
    })
    print(f"  {method:>6}: per-paragraph {ac_para:+.3f} | "
          f"arc real {ac_arc:+.3f} | arc shuffled {ac_shuf_mean:+.3f} | "
          f"reading-order excess {excess:+.3f}")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14.5, 7.2))
    draw_arc(a1, para_xy, f"Non-timeseries: per-paragraph (window = 1)\n"
                          f"{n} points, {method}")
    draw_arc(a2, arc_xy, f"Timeseries: windowed arc (size {args.size})\n"
                         f"{len(windows)} windows, {method}")
    fig.suptitle(
      f"{method}: does the arc survive without the window?  "
      f"reading-order excess coherence = {excess:+.2f}\n"
      f"(step smoothness beyond a shuffled-order control: "
      f">0 means reading order genuinely shapes the arc)", fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    p = os.path.join(out_dir, f"arc_projection_{method.lower().replace('-', '')}.png")
    fig.savefig(p, dpi=200); plt.close(fig)
    print(f"          wrote {p}")

  with open(os.path.join(out_dir, "projection_coherence.json"), "w") as f:
    json.dump({"book": args.book, "model": args.model, "size": args.size,
               "stride": args.stride, "methods": summary}, f, indent=2)

  # Bar chart: reading-order excess coherence per projection.
  fig, ax = plt.subplots(figsize=(7.6, 4.8))
  names = [s["method"] for s in summary]
  vals = [s["reading_order_excess"] for s in summary]
  colors = ["#2a78d6" if v > 0.05 else "#d1495b" for v in vals]
  ax.bar(names, vals, color=colors, alpha=0.85)
  ax.axhline(0, color="0.4", lw=0.8)
  ax.set_ylabel("reading-order excess coherence\n(arc real - arc shuffled)")
  ax.set_title("How much of each projection's arc is the story, not the method?\n"
               "bar = smoothness reading order adds beyond a shuffled control",
               fontsize=10.5)
  for i, v in enumerate(vals):
    ax.annotate(f"{v:+.2f}", (i, v), ha="center",
                va="bottom" if v >= 0 else "top", fontsize=10)
  fig.tight_layout()
  p = os.path.join(out_dir, "arc_projection_coherence.png")
  fig.savefig(p, dpi=200); plt.close(fig)
  print(f"\n  wrote {p}\n  -> {out_dir}")


if __name__ == "__main__":
  main()
