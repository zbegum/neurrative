"""
The geodesic plot, interactive: rotate, zoom, and hover to read the paragraphs.

Same landscape and same geodesics as geodesic_arrows.py -- this only changes the
renderer, so the two cannot drift apart. What the HTML adds is the thing a PNG
cannot do: turn the surface to see whether a path really goes *around* a hill,
and hover a curve to read the two paragraphs whose scores it connects.

Legend entries toggle: click "paragraphs" to clear the point cloud and see the
geodesics alone, or click the surface off to see the routes from below.

Example:

python arc_on_surface/geodesic_interactive.py --model bge-m3 --emotions humor
python arc_on_surface/geodesic_interactive.py --model bge-m3
"""

import argparse
import os
import sys

import numpy as np
import plotly.graph_objects as go


# Aliased: "paths" is already a local here (the geodesic polylines), and a
# module of that name would be shadowed inside main().
# The repository root and common/ on the path, wherever this script lives.
_ROOT = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_ROOT, "common")):
  _ROOT = os.path.dirname(_ROOT)
sys.path[1:1] = [_ROOT, os.path.join(_ROOT, "common")]

import paths as out_paths
from smooth_common import add_common_args, prepare
from data import load_paragraphs
from geodesic_arrows import build_surface, geodesics
from geometry.geodesic import polyline_deviation, polyline_length, rising_pairs



def short(text, chars=220):
  text = " ".join(text.split())
  return text if len(text) <= chars else text[:chars] + " ..."


def path_traces(paths, idx, deltas, scores, paragraphs, order, alpha):
  """One trace for every geodesic, joined by None gaps.

  348 separate traces would make the page crawl; a single trace with a break
  between each path draws them all at once and still hovers per path.
  """
  xs, ys, zs, text = [], [], [], []
  hx, hy, hz, htext = [], [], [], []

  for k in order:
    p = paths[k]
    if p is None or len(p) < 2:
      continue
    i = int(idx[k])
    z = p[:, 2] / alpha if alpha > 0 else np.zeros(len(p))
    label = (f"paragraph {i} → {i + 1}<br>"
             f"score {scores[i]:.2f} → {scores[i + 1]:.2f} "
             f"(+{deltas[k]:.2f})<br>"
             f"bend {polyline_deviation(p):.2f} · "
             f"length {polyline_length(p):.2f}")

    xs.extend(p[:, 0]); xs.append(None)
    ys.extend(p[:, 1]); ys.append(None)
    zs.extend(z);       zs.append(None)
    text.extend([label] * len(p)); text.append(None)

    hx.append(p[-1, 0]); hy.append(p[-1, 1]); hz.append(z[-1])
    htext.append(label + "<br><br><i>" + short(paragraphs[i + 1]["text"], 160) + "</i>")

  return [
    go.Scatter3d(
      x=xs, y=ys, z=zs, mode="lines", name=f"geodesics ({len(order)})",
      line=dict(color="#e34948", width=3), text=text, hoverinfo="text",
    ),
    go.Scatter3d(
      x=hx, y=hy, z=hz, mode="markers", name="arrow heads (i+1)",
      marker=dict(size=3, color="#e34948"), text=htext, hoverinfo="text",
    ),
  ]


def main():
  parser = add_common_args(argparse.ArgumentParser())
  parser.add_argument("--hx", default=0.2, type=float)
  parser.add_argument("--hy", default=0.2, type=float)
  parser.add_argument("--alpha", default=1.0, type=float,
                      help="Vertical exaggeration of the landscape.")
  parser.add_argument("--top", default=60, type=int,
                      help="How many of the strongest rises to draw.")
  parser.add_argument("--min-delta", default=0.0, type=float)
  args = parser.parse_args()

  ctx = prepare(args)
  paragraphs = load_paragraphs(args.book)
  out_dir = out_paths.out_dir(args.book, args.model, out_paths.GEODESICS)
  out_paths.stamp(out_dir, __file__, args, stack="geometry.smoothers",
                  estimator="gaussian_nw (anisotropic hx, hy; standardized coords)",
                  emotions=ctx["selected"])

  for emotion in ctx["selected"]:
    y_raw = ctx["matrix"][:, ctx["emotions"].index(emotion)]
    X_raw = ctx["X_raw"]
    print(f"\n{emotion}:")

    GX, GY, Z, mask = build_surface(X_raw, y_raw, args)
    idx, deltas = rising_pairs(y_raw, args.min_delta)
    _, _, paths, _ = geodesics(GX, GY, Z, mask, X_raw, idx, args.alpha)
    order = np.argsort(deltas)[::-1][:args.top]
    print(f"  {len(idx)} rising pairs, drawing {len(order)}")

    fig = go.Figure([
      go.Surface(
        x=GX, y=GY, z=Z, colorscale="Viridis", cmin=0.0, cmax=1.0,
        opacity=0.85, name="landscape", showscale=True,
        colorbar=dict(title=emotion, len=0.6),
        hovertemplate=f"{emotion} ≈ %{{z:.2f}}<extra></extra>",
      ),
      go.Scatter3d(
        x=X_raw[:, 0], y=X_raw[:, 1], z=y_raw, mode="markers",
        name="paragraphs", marker=dict(size=2, color="#33322e", opacity=0.5),
        text=[f"{p['id']} · chapter {p['chapter_id']}<br>"
              f"{emotion} {s:.2f}<br><br><i>{short(p['text'])}</i>"
              for p, s in zip(paragraphs, y_raw)],
        hoverinfo="text",
      ),
      *path_traces(paths, idx, deltas, y_raw, paragraphs, order, args.alpha),
    ])

    fig.update_layout(
      title=f"{emotion}: geodesics pointing toward {emotion} "
            f"(alpha={args.alpha:g}, {len(idx)} rising pairs, top {len(order)} drawn)",
      scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title=emotion,
                 zaxis=dict(range=[0, 1])),
      margin=dict(l=0, r=0, t=40, b=0), legend=dict(itemsizing="constant"),
    )

    path = os.path.join(out_dir, f"interactive_{emotion}.html")
    fig.write_html(path, include_plotlyjs="cdn")
    print(f"  wrote {path}")

  print(f"\nDone.\n{out_dir}")


if __name__ == "__main__":
  main()
