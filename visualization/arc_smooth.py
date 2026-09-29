"""
Smooth the narrative arc with distance-based curve smoothing (Pawellek et al.
2024). This replaces the earlier control-point geodesic fit (the retired
arc_geodesic.py): closeness to the arc is now one continuous parameter tau with a
provable bound, not a hand-tuned count of control points.

The narrative arc is a noisy polyline in the PCA (semantic) plane -- exactly the
"initial curve" the paper smooths. geometry/curve_smoothing.py implements their
method for our flat-plane case (see that module's docstring). Here we:

  * build the initial arc from the windowed embeddings,
  * smooth it across a sweep of the tolerance parameter tau (their Figure 5),
  * draw the sweep over the mood/emotion landscape (top view), and
  * lift one chosen smoothing onto the emotion terrain in 3-D.

tau = 0 gives the plain geodesic of the plane (a straight line between the book's
opening and closing); larger tau holds the smoothed curve closer to the original
arc. Unlike the old control-point fit, closeness is one continuous knob, not a
count of hand-placed points, and the result provably stays near the arc.

Example:

  python visualization/arc_smooth.py --book alice_wonderland --model bge-m3
  python visualization/arc_smooth.py --emotion wonder --taus 0 0.1 0.3 0.8
"""

import argparse
import os
from types import SimpleNamespace

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np

import paths
from data import load_paragraphs, load_scores
from smooth_common import load_pca
from arc_on_surface import window_bounds, build_surface
from arc_emotion_axis import (sample_surface, normalizer, mood, nan_blur,
                              DEFAULT_ORDER)
from geometry.curve_smoothing import smooth_curve


def _medoid(pts):
  """The most central actual point of a window: the one minimising total
  distance to the others. Unlike the mean it is a real paragraph, so it lies
  where the data (and hence the surface) actually is."""
  if len(pts) == 1:
    return pts[0]
  d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2).sum(axis=1)
  return pts[int(d.argmin())]


def build_emotion_surface(X_raw, y, args):
  """A single emotion's landscape (score as height) over the PCA grid."""
  GX, GY, Z, _xs, _ys = build_surface(X_raw, y, args.hx, args.hy, args.resolution,
                                      args.margin, args.density_floor)
  return GX, GY, Z


def build_mood_surface(X_raw, matrix, emotions, order, positions, args):
  """The combined mood field over the PCA grid (as in arc_emotion_axis)."""
  grid_Z, tf = [], {}
  GX = GY = None
  for e in order:
    y = matrix[:, emotions.index(e)]
    GX, GY, Z, _xs, _ys = build_surface(X_raw, y, args.hx, args.hy, args.resolution,
                                        args.margin, args.density_floor)
    grid_Z.append(Z)
    tf[e] = normalizer(y, args.norm)
  grid_n = np.stack([np.where(np.isnan(Z), np.nan, tf[e](Z))
                     for e, Z in zip(order, grid_Z)])
  supported = np.all(np.isfinite(grid_n), axis=0)
  Zmood = np.where(supported, mood(grid_n, positions, args.temp), np.nan)
  return GX, GY, nan_blur(Zmood, args.surface_smooth)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--book", default="alice_wonderland")
  ap.add_argument("--model", default="bge-m3")
  ap.add_argument("--emotion", default=None,
                  help="Draw over one emotion's landscape; default combined mood.")
  ap.add_argument("--size", default=10, type=int)
  ap.add_argument("--stride", default=5, type=int)
  ap.add_argument("--taus", nargs="+", type=float,
                  default=[0.0, 0.3, 0.6, 1.0, 1.6],
                  help="Tolerance sweep: 0 = straight geodesic, large = hugs arc.")
  ap.add_argument("--lift-tau", default=0.6, type=float,
                  help="Which tau to lift onto the terrain in the 3-D view.")
  ap.add_argument("--representative", default="medoid", choices=["medoid", "mean"],
                  help="Per-window point: 'medoid' is the most central real "
                       "paragraph (always in a dense region, so it lands on the "
                       "surface); 'mean' can fall in a void between clusters.")
  # penalty / solver knobs (sensible defaults; rarely need changing)
  ap.add_argument("--resolution", default=260, type=int)
  ap.add_argument("--blur", default=1.2, type=float)
  # surface (drawing) knobs -- same protocol as arc_on_surface
  ap.add_argument("--hx", default=0.15, type=float)
  ap.add_argument("--hy", default=0.15, type=float)
  ap.add_argument("--temp", default=0.3, type=float)
  ap.add_argument("--norm", default="rank", choices=["rank", "zscore", "minmax"])
  ap.add_argument("--surface-smooth", default=1.5, type=float)
  ap.add_argument("--margin", default=0.05, type=float)
  ap.add_argument("--density-floor", default=5.0, type=float)
  ap.add_argument("--alpha", default=3.0, type=float,
                  help="Vertical exaggeration of the terrain in the 3-D view.")
  ap.add_argument("--order", nargs="+", default=None)
  args = ap.parse_args()

  paragraphs = load_paragraphs(args.book)
  emotions, matrix = load_scores(args.book, paragraphs)
  X_raw = np.asarray(load_pca(args.book, args.model), dtype=np.float64)[:, :2]
  n = len(X_raw)
  order = args.order or [e for e in DEFAULT_ORDER if e in emotions]
  order += [e for e in emotions if e not in order]
  positions = np.linspace(0.0, 1.0, len(order))

  # The initial curve: one representative point per window, in the PCA plane.
  # The mean of a window can land in the empty gap between clusters (off the
  # surface); the medoid -- the window's most central actual paragraph -- is a
  # real point in a dense region, so it always sits on the surface.
  windows = window_bounds(n, args.size, args.stride)
  if args.representative == "mean":
    arc = np.array([X_raw[s:e].mean(axis=0) for s, e in windows])
  else:
    arc = np.array([_medoid(X_raw[s:e]) for s, e in windows])

  label = args.emotion or "mood"
  print(f"=== distance-based arc smoothing on the {label} surface: "
        f"{args.book} / {args.model} ===")
  print(f"  {n} paragraphs -> {len(windows)} windows (size {args.size} "
        f"stride {args.stride}); tau sweep {args.taus}")

  # The landscape, only for drawing. sfields carries what build_*_surface needs.
  sargs = SimpleNamespace(hx=args.hx, hy=args.hy, temp=args.temp, norm=args.norm,
                          surface_smooth=args.surface_smooth,
                          resolution=args.resolution, margin=args.margin,
                          density_floor=args.density_floor)
  if args.emotion:
    GX, GY, Zsurf = build_emotion_surface(
      X_raw, matrix[:, emotions.index(args.emotion)], sargs)
    zlabel = args.emotion
  else:
    GX, GY, Zsurf = build_mood_surface(X_raw, matrix, emotions, order, positions, sargs)
    zlabel = "mood"

  # Smooth the arc for each tau.
  results = {}
  for tau in args.taus:
    r = smooth_curve(arc, tau, resolution=args.resolution, blur=args.blur)
    results[tau] = r
    print(f"  tau={tau:<5}: length {r['length']:.3f} (arc {r['length0']:.3f}), "
          f"max deviation {r['max_dev']:.4f}")

  # The terrain the arc is smoothed over depends on the window, the stride and
  # the bandwidth, and on which emotion (or the mood blend) supplies the height
  # -- none of which the filenames beyond the tag record.
  tag = args.emotion or "mood"
  out_dir = paths.out_dir(args.book, args.model, os.path.join(paths.NARRATIVE_ARC_3D, paths.ARC_CURVE_SMOOTHING),
                          {"w": args.size, "s": args.stride, "h": args.hx,
                           "z": tag})
  paths.stamp(out_dir, __file__, args, stack="geometry.curve_smoothing",
              estimator="smooth_curve (distance-based, Pawellek 2024)",
              surface="gaussian_nw (geometry.smoothers) for the drawn terrain")

  # --- figure 1: the tau sweep over the landscape (top view) -------------------
  fig, ax = plt.subplots(figsize=(9, 8.2))
  cf = ax.contourf(GX, GY, Zsurf, levels=18, cmap="magma", alpha=0.85)
  fig.colorbar(cf, ax=ax, shrink=0.7, label=zlabel)
  ax.plot(arc[:, 0], arc[:, 1], color="0.25", lw=1.0, alpha=0.7,
          label="narrative arc (initial)", zorder=3)
  ramp = plt.get_cmap("winter")
  for i, tau in enumerate(args.taus):
    c = results[tau]["curve"]
    col = ramp(i / max(1, len(args.taus) - 1))
    ax.plot(c[:, 0], c[:, 1], color=col, lw=2.4, zorder=4,
            label=f"tau = {tau:g}")
  ax.scatter(*arc[0], color="black", s=90, marker="o", zorder=6)
  ax.scatter(*arc[-1], color="black", s=110, marker="X", zorder=6)
  ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
  ax.set_aspect("equal", adjustable="box")
  ax.legend(loc="best", fontsize=8, framealpha=0.9)
  ax.set_title(f"Distance-based smoothing of the narrative arc (Pawellek 2024)\n"
               f"tau: 0 = straight geodesic -> large = hugs the arc  |  "
               f"{label} landscape", fontsize=11)
  fig.tight_layout()
  p = os.path.join(out_dir, f"arc_smooth_{tag}_sweep.png")
  fig.savefig(p, dpi=170); plt.close(fig); print(f"  wrote {p}")

  # --- figure 2: one smoothing lifted onto the terrain -------------------------
  lift_tau = min(args.taus, key=lambda t: abs(t - args.lift_tau))
  smooth = results[lift_tau]["curve"]

  # Unproject onto the surface: read the surface height at each point. Points
  # over a masked (low-density) hole get no height from interpolation, so we
  # snap those to the nearest supported surface cell -- every point lands ON the
  # surface rather than floating.
  from scipy.spatial import cKDTree
  smask = np.isfinite(Zsurf)
  svalid_xy = np.column_stack([GX[smask], GY[smask]])
  svalid_z = Zsurf[smask]
  stree = cKDTree(svalid_xy)

  def lift(xy):
    z = sample_surface(GX, GY, Zsurf, xy[:, 0], xy[:, 1])
    gap = ~np.isfinite(z)
    if gap.any():
      z[gap] = svalid_z[stree.query(xy[gap])[1]]
    return np.column_stack([xy[:, 0], xy[:, 1], args.alpha * z])

  arc3d, smooth3d = lift(arc), lift(smooth)
  lz = 0.03 * args.alpha                          # small lift so lines clear the surface
  fig = plt.figure(figsize=(12, 9))
  ax3 = fig.add_subplot(111, projection="3d")
  # muted grey terrain so the progression-coloured curve is the focus. Draw at
  # full resolution (stride 1) so the drawn surface matches where the dots are
  # sampled from -- a coarser stride would leave the dots hovering over quads
  # that don't reach them.
  ax3.plot_surface(GX, GY, args.alpha * Zsurf, color="0.7", linewidth=0,
                   antialiased=True, alpha=0.25, rstride=1, cstride=1,
                   shade=True)
  # the timeseries points we smooth over -- small dots, no connecting edges,
  # coloured by reading progression (dark = start, bright = end).
  prog = np.linspace(0, 1, len(arc3d))
  ax3.scatter(arc3d[:, 0], arc3d[:, 1], arc3d[:, 2] + lz, c=prog, cmap="plasma",
              s=9, depthshade=False, edgecolors="none", zorder=6)
  # the smoothed curve -- thin, coloured by the same progression.
  seg = np.stack([smooth3d[:-1], smooth3d[1:]], axis=1)
  lc = Line3DCollection(seg + [0, 0, lz], cmap="plasma", linewidth=1.6, zorder=7)
  lc.set_array((prog[:-1] + prog[1:]) / 2)
  ax3.add_collection3d(lc)
  ax3.set_xlabel("PC1"); ax3.set_ylabel("PC2"); ax3.set_zlabel(zlabel)
  ax3.set_xticklabels([]); ax3.set_yticklabels([])
  ax3.view_init(elev=32, azim=-52)
  ax3.set_title(f"Smoothed narrative arc on the {label} terrain "
                f"(tau = {lift_tau:g})", fontsize=12)
  fig.tight_layout()
  p = os.path.join(out_dir, f"arc_smooth_{tag}_3d.png")
  fig.savefig(p, dpi=170); plt.close(fig); print(f"  wrote {p}")

  # --- interactive: rotatable terrain with a tau slider ------------------------
  # Drag tau from 0 (straight geodesic) upward and watch the curve progressively
  # hug the arc; rotate to see it ride the landscape.
  try:
    import plotly.graph_objects as go

    lifted = {t: lift(results[t]["curve"]) for t in args.taus}
    prog = np.linspace(0, 1, len(arc3d))
    traces = [
      go.Surface(x=GX, y=GY, z=args.alpha * Zsurf, colorscale="Greys",
                 cmin=-args.alpha, cmax=2 * args.alpha, opacity=0.45,
                 showscale=False, hoverinfo="skip", name=zlabel),
      # the timeseries points we smooth over: small dots, no edges, coloured by
      # reading progression (dark = start, bright = end).
      go.Scatter3d(x=arc3d[:, 0], y=arc3d[:, 1], z=arc3d[:, 2] + 0.02,
                   mode="markers",
                   marker=dict(size=2.5, color=prog, colorscale="Plasma"),
                   name="sample points", hovertemplate="window<extra></extra>"),
    ]
    # one smoothed-curve trace per tau; only the lift_tau one visible at first
    for t in args.taus:
      c3 = lifted[t]
      cp = np.linspace(0, 1, len(c3))
      traces.append(go.Scatter3d(
        x=c3[:, 0], y=c3[:, 1], z=c3[:, 2] + 0.03 * args.alpha, mode="lines",
        line=dict(color=cp, colorscale="Plasma", width=4),
        name=f"smoothed (tau={t:g})",
        visible=(t == lift_tau), hovertemplate=f"tau={t:g}<extra></extra>"))
    steps = []
    for i, t in enumerate(args.taus):
      vis = [True, True] + [j == i for j in range(len(args.taus))]
      steps.append(dict(method="update", label=f"{t:g}",
                        args=[{"visible": vis}]))
    figi = go.Figure(traces)
    figi.update_layout(
      title=f"Distance-based smoothing of the narrative arc -- {label} terrain "
            f"(drag tau; rotate to see the arc ride the landscape)",
      sliders=[dict(active=args.taus.index(lift_tau), currentvalue={"prefix": "tau = "},
                    pad={"t": 40}, steps=steps)],
      scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title=zlabel,
                 xaxis=dict(showticklabels=False), yaxis=dict(showticklabels=False),
                 camera=dict(eye=dict(x=1.5, y=-1.6, z=0.9))),
      height=800, margin=dict(l=0, r=0, t=50, b=0))
    p = os.path.join(out_dir, f"arc_smooth_{tag}.html")
    figi.write_html(p, include_plotlyjs="cdn"); print(f"  wrote {p}")
  except ImportError:
    print("  (plotly not available; skipped interactive HTML)")

  print(f"  -> {out_dir}")


if __name__ == "__main__":
  main()
