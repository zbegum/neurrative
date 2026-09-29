"""
Interactive (rotatable) version of the smoother x bandwidth mood-surface grid.

Same content as arc_emotion_smoother_grid.py -- rows are the four smoothers,
columns their bandwidth ladder, each cell the combined mood surface with the arc
riding it -- but rendered as a Plotly grid of 3-D scenes you can spin, zoom, and
hover. Every scene keeps the emotion-labelled z-axis, so height still reads as
mood. Hovering the arc shows the reading position and mood; hovering the surface
shows the mood at that point on the landscape.

Writes a single self-contained HTML to
arc_on_surface/output/<book>/<model>/arc_on_surface/<variant>/emotion_axis_smoother_grid.html,
sharing the static twin's variant so the HTML sits beside its PNG.

loess/local_linear fit a regression per grid point, so this is slow -- run it in
the background.

Example:

  python arc_on_surface/arc_emotion_smoother_grid_interactive.py --book alice_wonderland --model bge-m3
"""

import argparse
import os
import sys

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# The repository root and common/ on the path, wherever this script lives.
_ROOT = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_ROOT, "common")):
  _ROOT = os.path.dirname(_ROOT)
sys.path[1:1] = [_ROOT, os.path.join(_ROOT, "common")]

import paths
from data import load_paragraphs, load_scores
from smooth_common import load_pca
from arc_on_surface import window_bounds
from arc_emotion_axis import (normalizer, mood, smooth1d, nan_blur,
                              sample_surface, EMOTION_COLOR, DEFAULT_ORDER)
from arc_emotion_smoother_grid import (METHODS, fit_surface, surface_at,
                                       variant_params)


def scene_key(idx):
  """Plotly names the first scene 'scene', then 'scene2', 'scene3', ..."""
  return "scene" if idx == 0 else f"scene{idx + 1}"


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--book", default="alice_wonderland")
  ap.add_argument("--model", default="bge-m3")
  ap.add_argument("--size", default=10, type=int)
  ap.add_argument("--stride", default=5, type=int)
  ap.add_argument("--temp", default=0.3, type=float)
  ap.add_argument("--norm", default="rank", choices=["rank", "zscore", "minmax"])
  ap.add_argument("--arc-smooth", default=3.0, type=float)
  ap.add_argument("--surface-smooth", default=1.5, type=float)
  ap.add_argument("--resolution", default=60, type=int)
  ap.add_argument("--margin", default=0.05, type=float)
  ap.add_argument("--density-floor", default=5.0, type=float)
  ap.add_argument("--order", nargs="+", default=None)
  args = ap.parse_args()

  paragraphs = load_paragraphs(args.book)
  emotions, matrix = load_scores(args.book, paragraphs)
  X_raw = np.asarray(load_pca(args.book, args.model), dtype=np.float64)[:, :2]
  n = len(X_raw)
  order = args.order or [e for e in DEFAULT_ORDER if e in emotions]
  order += [e for e in emotions if e not in order]
  positions = np.linspace(0.0, 1.0, len(order))
  tf = {e: normalizer(matrix[:, emotions.index(e)], args.norm) for e in order}

  windows = window_bounds(n, args.size, args.stride)
  centers = np.array([(s + e - 1) / 2.0 for s, e in windows])
  arc_xy = np.array([X_raw[s:e].mean(axis=0) for s, e in windows])

  # Borrowed from the static twin so the HTML lands in the same variant
  # directory as the PNG it mirrors.
  out_dir = paths.out_dir(args.book, args.model, os.path.join(paths.NARRATIVE_ARC_3D, paths.ARC_MOOD_AXIS),
                          variant_params(args))
  paths.stamp(out_dir, __file__, args, stack="geometry.smoothers",
              estimator="swept: " + ", ".join(name for name, *_ in METHODS),
              spectrum=order)
  nr, nc = len(METHODS), 3
  print(f"=== interactive smoother grid: {args.book} / {args.model} ===")
  print(f"  {nr} smoothers x {nc} bandwidths, res {args.resolution}")

  titles = [f"{name}  {lab}" for name, _, _, labs in METHODS for lab in labs]
  fig = make_subplots(
    rows=nr, cols=nc, subplot_titles=titles,
    specs=[[{"type": "scene"} for _ in range(nc)] for _ in range(nr)],
    horizontal_spacing=0.02, vertical_spacing=0.05)

  scene_layouts = {}
  for r, (name, predict, param_cols, labels) in enumerate(METHODS):
    for c, params in enumerate(param_cols):
      grid_Z, arc_h = [], []
      GX = GY = None
      for e in order:
        y = matrix[:, emotions.index(e)]
        GX, GY, Z, xs, ys = fit_surface(predict, params, X_raw, y,
                                        args.resolution, args.margin, args.density_floor)
        grid_Z.append(Z)
        arc_h.append(surface_at(predict, params, xs, ys, X_raw, y, arc_xy))
      grid_n = np.stack([np.where(np.isnan(Z), np.nan, tf[e](Z))
                         for e, Z in zip(order, grid_Z)])
      arc_n = np.stack([tf[e](hh) for e, hh in zip(order, arc_h)])
      all_supported = np.all(np.isfinite(grid_n), axis=0)
      mood_grid = np.where(all_supported, mood(grid_n, positions, args.temp), np.nan)
      mood_grid = nan_blur(mood_grid, args.surface_smooth)
      mood_arc = mood(arc_n, positions, args.temp)
      dom = np.array(order)[np.nanargmax(np.where(np.isfinite(arc_n), arc_n, -np.inf), axis=0)]

      ax = smooth1d(arc_xy[:, 0], args.arc_smooth)
      ay = smooth1d(arc_xy[:, 1], args.arc_smooth)
      az_on = sample_surface(GX, GY, mood_grid, ax, ay)
      az = np.where(np.isfinite(az_on), az_on, smooth1d(mood_arc, args.arc_smooth)) + 0.01
      t = np.linspace(0, 1, len(ax))
      arc_text = [f"paragraph ~{int(centers[i])}<br>mood {mood_arc[i]:.2f}<br>"
                  f"dominant {dom[i]}" for i in range(len(ax))]

      fig.add_trace(go.Surface(
        x=GX, y=GY, z=mood_grid, colorscale="Magma", cmin=0, cmax=1,
        opacity=0.6, showscale=False, hovertemplate="mood %{z:.2f}<extra></extra>"),
        row=r + 1, col=c + 1)
      fig.add_trace(go.Scatter3d(
        x=ax, y=ay, z=az, mode="lines",
        line=dict(color=t, colorscale="Plasma", width=5),
        text=arc_text, hoverinfo="text", showlegend=False),
        row=r + 1, col=c + 1)
      fig.add_trace(go.Scatter3d(
        x=ax[::4], y=ay[::4], z=az[::4], mode="markers",
        marker=dict(size=3, color=[EMOTION_COLOR.get(e, "#555") for e in dom[::4]]),
        text=[arc_text[i] for i in range(0, len(ax), 4)], hoverinfo="text",
        showlegend=False), row=r + 1, col=c + 1)
      fig.add_trace(go.Scatter3d(
        x=[ax[0], ax[-1]], y=[ay[0], ay[-1]], z=[az[0], az[-1]], mode="markers",
        marker=dict(size=5, color="black", symbol=["circle", "x"]),
        text=["opening (O)", "ending (X)"], hoverinfo="text", showlegend=False),
        row=r + 1, col=c + 1)

      scene_layouts[scene_key(r * nc + c)] = dict(
        xaxis=dict(title="PC1", showticklabels=False),
        yaxis=dict(title="PC2", showticklabels=False),
        zaxis=dict(tickvals=positions, ticktext=order, range=[0, 1]),
        camera=dict(eye=dict(x=1.5, y=-1.6, z=0.9)),
        aspectmode="cube")
      print(f"    {name} {labels[c]} done")

  fig.update_layout(
    title=f"Emotion-axis mood surface by smoother and bandwidth -- "
          f"{args.book} / {args.model}  (drag to rotate; hover the arc)",
    height=430 * nr, width=1500, margin=dict(l=10, r=10, t=70, b=10),
    **scene_layouts)
  for a in fig.layout.annotations:      # shrink the subplot titles
    a.font.size = 11

  p = os.path.join(out_dir, "emotion_axis_smoother_grid.html")
  fig.write_html(p, include_plotlyjs="cdn")
  print(f"  wrote {p}")


if __name__ == "__main__":
  main()
