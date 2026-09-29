"""
The arc on the surface, interactive: rotate, zoom, and hover to read the windows.

Same curve and same terrain as arc_on_surface.py -- this only changes the
renderer, so the two cannot drift apart. What the HTML adds is what the PNG
cannot show: the arc doubles back on itself constantly (157 windows crowded into
a small region of PCA space), so from one fixed angle it reads as a tangle. Turn
it and the route separates.

Four toggleable layers per emotion, so the comparison can be taken apart:

  terrain          the fitted emotion surface.
  route            the dense lift -- the curve whose shadow is the 2-D arc, and
                   the only one that actually lies on the surface.
  the floor        the PCA plane itself at z = 0: all 789 paragraphs as a grey
                   cloud -- the points the surface is fitted over, and the only
                   place in this figure they appear -- with the 157 time-window
                   means on top of them and the 2-D narrative arc through those.
                   That arc is the route's own shadow, so showing both at once
                   makes the definition of the lift visible rather than asserted.
  windows          the 157 real windows, as points on that route, hoverable:
                   reading position, surface height, the window's own emotion,
                   and the gap between them. Points and not a polyline -- the
                   chords between consecutive windows leave the surface by up to
                   0.146 (a third of the relief), which is exactly the curve
                   this file exists not to draw.

  own emotion      the same 157 windows at the height the *text* carried. Where
                   this sits above the route, the story felt more than its
                   location predicts.

One colour scale on the page, and it is not the emotion:

  grey (the terrain)   the emotion -- but height says that already, so it is
                       drawn without a scale of its own, as arc_smooth.py does.
  plasma (the route)   *reading position*, in paragraphs. Dark = the opening,
                       bright = the last page. Same ramp as the B-spline
                       figures. The arc doubles back so often that nothing else
                       says which way the story is running through the tangle.

One emotion per file, because six rotatable surfaces in one page is slow and
nothing is gained by seeing them at once -- the comparison that matters is
between the two curves, not between emotions.

Example:

python arc_on_surface/arc_on_surface_interactive.py --model bge-m3 --emotions wonder
python arc_on_surface/arc_on_surface_interactive.py --model bge-m3
"""

import argparse
import json
import os
import sys

import numpy as np
import plotly.graph_objects as go


# The repository root and common/ on the path, wherever this script lives.
_ROOT = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_ROOT, "common")):
  _ROOT = os.path.dirname(_ROOT)
sys.path[1:1] = [_ROOT, os.path.join(_ROOT, "common")]

import paths
from data import load_paragraphs, load_scores
from smooth_common import load_pca
from arc_on_surface import (build_surface, geodesic_route, resample,
                            support_at, surface_at, window_bounds,
                            window_points)

# Matched to the PNG so the two renderers read as one figure.
ROUTE = "#d1495b"
OWN = "#1baf7a"
UNSUPPORTED = "rgba(140,140,140,0.9)"


def figure(GX, GY, Z, X_raw, dense_xy, z_dense, dense_t, supported, arc_xy,
           z_win, own_win, centers, windows, summaries, emotion, book, model,
           args, runs=None):
  fig = go.Figure()

  # Grey, following arc_smooth.py: the terrain's colour is redundant with its
  # height (both are the emotion, and the z axis is labelled with it), and
  # spending a second warm ramp on it is what makes the plasma route vanish into
  # the hillside. One colour scale on the page, and it means reading position.
  fig.add_surface(
    x=GX, y=GY, z=Z, colorscale="Greys", cmin=-0.15, cmax=1.15, opacity=0.55,
    showscale=False, name="terrain", showlegend=True,
    hovertemplate=f"PC1 %{{x:.3f}}<br>PC2 %{{y:.3f}}<br>{emotion} %{{z:.3f}}"
                  "<extra>terrain</extra>",
  )

  # The dense lift, split the same way arc_on_surface.py measures it: coloured
  # where the surface is supported, grey where the height is extrapolated across
  # a hole in the terrain. Drawn without hover -- 6500 points, none of which is a
  # window, so a tooltip there would invite reading interpolation as data.
  def broken(mask):
    """The dense curve with the points outside `mask` lifted out as gaps."""
    keep = np.where(mask, 1.0, np.nan)
    return (dense_xy[:, 0] * keep, dense_xy[:, 1] * keep, z_dense * keep)

  # Colour is reading position in *paragraphs*, not an abstract 0..1, so the
  # colorbar can be read against the book itself. It is the only scale on the
  # page, and it is time -- the emotion is the height.
  pos = np.interp(dense_t, np.arange(len(centers)), centers)
  span = dict(cmin=float(centers[0]), cmax=float(centers[-1]))

  bar = dict(title="reading<br>position", len=0.55, thickness=14)

  if runs is not None:
    # Geodesic legs. Every point is on the mesh, so there is nothing to split
    # into supported and extrapolated -- the route cannot leave the terrain.
    # NaN between runs breaks the line where a leg could not be solved, rather
    # than bridging it with a segment nobody computed.
    gx, gy, gz, gc = [], [], [], []
    for P, tt in runs:
      gx += list(P[:, 0]) + [np.nan]
      gy += list(P[:, 1]) + [np.nan]
      gz += list(P[:, 2] + args.draw_lift) + [np.nan]
      gc += list(np.interp(tt, np.arange(len(centers)), centers)) + [np.nan]
    fig.add_scatter3d(
      x=gx, y=gy, z=gz, mode="lines",
      line=dict(color=gc, colorscale="Plasma", width=5, showscale=True,
                colorbar=bar, **span),
      name="route (geodesic legs)", hoverinfo="skip",
    )
  else:
    x, y, zz = broken(supported)
    fig.add_scatter3d(
      x=x, y=y, z=zz, mode="lines",
      line=dict(color=pos, colorscale="Plasma", width=5, showscale=True,
                colorbar=bar, **span),
      name="route (on the surface)", hoverinfo="skip",
    )
  if runs is None and not supported.all():
    # Both endpoints of each gap are kept so the grey visibly bridges the hole
    # rather than vanishing into it.
    bridge = ~supported
    bridge[:-1] |= supported[:-1] & ~supported[1:]
    bridge[1:] |= supported[1:] & ~supported[:-1]
    x, y, zz = broken(bridge)
    fig.add_scatter3d(
      x=x, y=y, z=zz, mode="lines",
      line=dict(color=UNSUPPORTED, width=2),
      name="extrapolated (no terrain)", hoverinfo="skip",
    )

  gap = z_win - own_win
  text = [
    f"window {i} · paragraphs {s}–{e - 1}<br>"
    f"reading position {c:.0f}<br>"
    f"surface height {zw:.3f}<br>"
    f"window's own {emotion} {ow:.3f}<br>"
    f"gap {g:+.3f}<br><br>{sm}"
    for i, ((s, e), c, zw, ow, g, sm) in enumerate(
      zip(windows, centers, z_win, own_win, gap, summaries))
  ]

  # --- the floor: the PCA plane itself, at z = 0 -------------------------------
  # The surface is fitted over the 789 paragraphs, but until now none of them was
  # drawn -- only the 157 window means, and only up on the terrain. Putting the
  # cloud and the windows on the floor makes the plane readable as a plane: the
  # domain the surface interpolates over, the windows as averages of ten of those
  # points, and the arc as the path through them.
  fig.add_scatter3d(
    x=X_raw[:, 0], y=X_raw[:, 1], z=np.zeros(len(X_raw)), mode="markers",
    marker=dict(size=1.8, color="rgba(90,90,90,0.45)", line=dict(width=0)),
    name=f"paragraphs in PCA ({len(X_raw)})",
    text=[f"paragraph {i}" for i in range(len(X_raw))], hoverinfo="text",
  )

  # The shadow: the same route flattened onto the PCA plane at z = 0. This is
  # the 2-D narrative arc -- literally, since the lift is defined as the curve
  # whose vertical projection is that arc -- so showing both at once is what
  # makes the definition visible instead of asserted.
  fig.add_scatter3d(
    x=dense_xy[:, 0], y=dense_xy[:, 1], z=np.zeros(len(dense_xy)), mode="lines",
    line=dict(color=pos, colorscale="Plasma", width=2, showscale=False, **span),
    opacity=0.6, name="the arc, in the plane (z = 0)", hoverinfo="skip",
  )
  fig.add_scatter3d(
    x=arc_xy[:, 0], y=arc_xy[:, 1], z=np.zeros(len(arc_xy)), mode="markers",
    marker=dict(size=3.5, color=centers, colorscale="Plasma", showscale=False,
                line=dict(width=0), **span),
    name=f"time windows, in the plane ({len(arc_xy)})", text=text,
    hoverinfo="text",
  )

  # The stems, as one trace with a break between each so the page stays fast.
  sx, sy, sz = [], [], []
  for j in range(len(arc_xy)):
    sx += [arc_xy[j, 0], arc_xy[j, 0], None]
    sy += [arc_xy[j, 1], arc_xy[j, 1], None]
    sz += [z_win[j], own_win[j], None]
  fig.add_scatter3d(x=sx, y=sy, z=sz, mode="lines",
                    line=dict(color="rgba(120,120,120,0.55)", width=2),
                    name="the gap", hoverinfo="skip")

  # Markers, not a line. These 157 points sit exactly on the route -- they are
  # the windows it was built from -- but joining them with straight chords draws
  # a curve that does NOT lie on the surface: each chord spans about ten grid
  # cells of curved terrain and cuts through it by up to 0.146, a third of the
  # relief. That chord curve is the thing this file was rewritten to stop
  # drawing, so the points are shown as points and the route carries the line.
  fig.add_scatter3d(
    x=arc_xy[:, 0], y=arc_xy[:, 1], z=z_win, mode="markers",
    marker=dict(size=4, color=centers, colorscale="Plasma", showscale=False,
                line=dict(width=0), **span),
    name="windows (on the route)", text=text, hoverinfo="text",
  )
  fig.add_scatter3d(
    x=arc_xy[:, 0], y=arc_xy[:, 1], z=own_win, mode="lines+markers",
    line=dict(color=OWN, width=3),
    marker=dict(size=3, color=OWN),
    name="the window's own emotion", text=text, hoverinfo="text",
  )

  # Start and end, so the direction of reading is never ambiguous.
  fig.add_scatter3d(
    x=[arc_xy[0, 0], arc_xy[-1, 0]], y=[arc_xy[0, 1], arc_xy[-1, 1]],
    z=[z_win[0], z_win[-1]], mode="markers+text",
    marker=dict(size=6, color="black", symbol=["circle", "x"]),
    text=["start", "end"], textposition="top center",
    name="start / end", hoverinfo="skip",
  )

  fig.update_layout(
    title=(f"{emotion}: the narrative arc on the emotion surface — "
           f"{book} / {model}<br>"
           f"<sub>window {args.size}, stride {args.stride}, h = {args.hx} · "
           f"grey = the {emotion} terrain (height is the score) · "
           f"colour = reading position, paragraph "
           f"{centers[0]:.0f} → {centers[-1]:.0f} · "
           f"dots are the real windows; green is what the text carried · "
           f"{len(arc_xy)} windows, "
           f"{sum(len(P) for P, _ in runs) if runs else len(dense_xy)} "
           f"curve points</sub>"),
    scene=dict(xaxis_title="PC1", yaxis_title="PC2",
               zaxis_title=emotion, zaxis=dict(range=[0, 1]),
               aspectmode="cube"),
    legend=dict(orientation="h", y=-0.04),
    margin=dict(l=0, r=0, t=76, b=0), height=820,
  )
  return fig


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--book", default="alice_wonderland")
  ap.add_argument("--model", default="bge-m3")
  ap.add_argument("--size", default=10, type=int)
  ap.add_argument("--stride", default=5, type=int)
  ap.add_argument("--emotions", nargs="*", default=None,
                  help="Subset of emotions (default: all).")
  ap.add_argument("--hx", default=0.15, type=float)
  ap.add_argument("--hy", default=0.15, type=float)
  ap.add_argument("--alpha", default=1.0, type=float,
                  help="With --legs straight this only places the output beside "
                       "the PNGs from the same run; the HTML draws in score "
                       "units. With --legs geodesic it is the exchange rate "
                       "between score and PCA units used to mesh the surface, "
                       "so it decides what 'shortest' means.")
  ap.add_argument("--oversample", default=4.0, type=float)
  ap.add_argument("--draw-lift", default=0.008, type=float,
                  help="Cosmetic only: how far above the terrain the route is "
                       "drawn, in score units. A geodesic lies exactly on the "
                       "surface, so at 0 it z-fights with the semi-transparent "
                       "terrain and renders as a broken curve. Same trick as "
                       "arc_smooth.py. It does not affect any measurement.")
  ap.add_argument("--legs", default="straight", choices=["straight", "geodesic"],
                  help="How consecutive waypoints are joined. 'straight' is a "
                       "line in the PCA plane, lifted -- which crosses the gaps "
                       "between clusters. 'geodesic' is the shortest path along "
                       "the surface, which cannot leave it.")
  ap.add_argument("--snap-tol", default=1.0, type=float,
                  help="How far (in grid cells) a waypoint may sit off the mesh "
                       "and still anchor a geodesic leg.")
  ap.add_argument("--window-point", default="mean", choices=["mean", "medoid"],
                  help="What stands for a window: the centroid of its "
                       "paragraphs (mean), or the most central real paragraph "
                       "in it (medoid). The medoid is a place in the book; the "
                       "mean is not.")
  ap.add_argument("--resolution", default=120, type=int)
  ap.add_argument("--margin", default=0.05, type=float)
  ap.add_argument("--density-floor", default=5.0, type=float)
  args = ap.parse_args()

  paragraphs = load_paragraphs(args.book)
  emotions, matrix = load_scores(args.book, paragraphs)
  X_raw = np.asarray(load_pca(args.book, args.model), dtype=np.float64)[:, :2]
  n = len(X_raw)
  if len(paragraphs) != n:
    raise ValueError(f"pca ({n}) != paragraphs ({len(paragraphs)})")

  selected = args.emotions or emotions
  unknown = [e for e in selected if e not in emotions]
  if unknown:
    raise ValueError(f"Unknown emotion(s) {unknown}. Available: {emotions}")

  # Same directory as the PNGs from the same settings, so a run's static and
  # interactive views never separate.
  out_dir = paths.out_dir(
    args.book, args.model,
    os.path.join(paths.NARRATIVE_ARC_3D, paths.ARC_FROM_TERRAIN),
    {"w": args.size, "s": args.stride, "h": args.hx,
     "hy": args.hy if args.hy != args.hx else None,
     "a": args.alpha,
     "pt": None if args.window_point == "mean" else args.window_point,
     "legs": None if args.legs == "straight" else args.legs,
     "df": None if args.density_floor == 5.0 else args.density_floor})

  windows = window_bounds(n, args.size, args.stride)
  centers = np.array([(s + e - 1) / 2.0 for s, e in windows])
  arc_xy, arc_idx = window_points(X_raw, windows, args.window_point)

  # The annotator wrote a one-line summary per paragraph and nothing in this
  # repo has ever read it. It is the right hover anchor: a summary says what the
  # story is doing there far faster than raw text does.
  #
  # Which paragraph's summary depends on what the point is. For a medoid the
  # answer is exact -- the point *is* that paragraph, so its summary describes
  # the place you are hovering. For a mean there is no such paragraph, so the
  # window's first one stands in, and the tooltip says so rather than implying
  # the centroid has a text.
  with open(os.path.join(paths.ROOT, "books", args.book,
                         "paragraph_scores.json")) as f:
    scored = json.load(f)
  anchors = arc_idx if arc_idx is not None else [s for s, _ in windows]
  summaries = []
  for a in anchors:
    a = int(a)
    text = (scored[a].get("summary") if a < len(scored) else None) or \
           paragraphs[a].get("text", "")
    text = " ".join(str(text).split())
    text = text if len(text) <= 170 else text[:170] + " ..."
    label = (f"paragraph {a} (the medoid)" if arc_idx is not None
             else f"paragraph {a} (first in the window; the point is a centroid)")
    summaries.append(f"{label}<br>{text}")

  probe = build_surface(X_raw, matrix[:, 0], args.hx, args.hy,
                        args.resolution, args.margin, args.density_floor)
  cell = float(abs(probe[0][0, 1] - probe[0][0, 0]))
  dense_xy, dense_t = resample(arc_xy, cell / args.oversample)

  print(f"=== interactive arc on surface: {args.book} / {args.model} ===")
  print(f"  {len(windows)} windows, {len(dense_xy)} curve points\n")

  for e in selected:
    y_raw = matrix[:, emotions.index(e)]
    GX, GY, Z, xs, ys = build_surface(X_raw, y_raw, args.hx, args.hy,
                                      args.resolution, args.margin,
                                      args.density_floor)
    z_dense = surface_at(xs, ys, X_raw, y_raw, dense_xy, args.hx, args.hy)
    sup, floor = support_at(xs, ys, X_raw, y_raw, dense_xy, args.hx, args.hy,
                            args.density_floor)
    supported = sup >= floor
    z_win = surface_at(xs, ys, X_raw, y_raw, arc_xy, args.hx, args.hy)
    own_win = np.array([y_raw[s:e2].mean() for s, e2 in windows])

    runs = None
    if args.legs == "geodesic":
      runs, st = geodesic_route(GX, GY, Z, arc_xy, args.alpha,
                                args.snap_tol * cell)
      print(f"  {e:>10}: {st['legs_solved']}/{st['legs_total']} geodesic legs, "
            f"{st['runs']} unbroken run(s), "
            f"{st['waypoints_on_mesh']}/{len(arc_xy)} waypoints on the mesh")

    fig = figure(GX, GY, Z, X_raw, dense_xy, z_dense, dense_t, supported,
                 arc_xy, z_win, own_win, centers, windows, summaries, e,
                 args.book, args.model, args, runs)
    p = os.path.join(out_dir, paths.named("arc_on_surface", e, "html"))
    fig.write_html(p, include_plotlyjs="cdn")
    print(f"  {e:>10}: gap {np.abs(z_win - own_win).mean():.3f}  -> {p}")

  paths.stamp(out_dir, __file__, args, stack="geometry.smoothers",
              estimator="gaussian_nw (anisotropic hx, hy; standardized coords)",
              renderer="plotly", emotions=selected)
  print(f"\nDone.\n{out_dir}")


if __name__ == "__main__":
  main()
