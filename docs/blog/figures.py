"""Figures for the narrative-arc blog post (Alice, bge-m3, PCA plane).

Needs projection/embedding.py to have written pca.npy.

python docs/blog/figures.py
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[1:1] = [ROOT, os.path.join(ROOT, "common"), os.path.join(ROOT, "arc")]

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np

from narrative_arc.curves import FitOpts, fit_curve
from narrative_arc.windows import window_bounds
from surface.mood import data as mood_data, surface as mood_surface
from surface.mood.mood import EMOTION_COLOR, SPECTRUM, mood
from surface.mood.geodesic import route as mood_route

BOOK, MODEL = "alice_wonderland", "bge-m3"
SIZE, STRIDE = 40, 20
TIME = plt.get_cmap("plasma")
INK, FAINT = "#2b2b2b", "#c9c8c2"
MOOD = LinearSegmentedColormap.from_list(
  "mood", list(zip(np.linspace(0, 1, len(SPECTRUM)), [EMOTION_COLOR[e] for e in SPECTRUM])))

plt.rcParams.update({
  "font.family": "sans-serif", "font.size": 10, "axes.edgecolor": FAINT,
  "axes.labelcolor": INK, "xtick.color": "#8a8983", "ytick.color": "#8a8983",
  "savefig.facecolor": "white", "savefig.bbox": "tight", "savefig.dpi": 200,
})


def windows(X, size, stride):
  """Window means in the plane: PCA is linear, so this is the projected mean."""
  return np.array([X[a:b].mean(axis=0) for a, b in window_bounds(len(X), size, stride)])


def draped(ax, points, lw, lift, alpha=1.0, samples=8):
  """A line through the plane points, sampled densely and lifted onto the surface."""
  dense = np.concatenate([np.linspace(points[i], points[i + 1], samples, endpoint=False)
                          for i in range(len(points) - 1)] + [points[-1:]])
  P3 = np.column_stack([dense, lift(dense) + 0.004])
  seg = np.stack([P3[:-1], P3[1:]], axis=1)
  lc = Line3DCollection(seg, cmap=TIME, linewidths=lw, alpha=alpha, capstyle="round")
  lc.set_array(np.linspace(0, 1, len(seg)))
  ax.add_collection3d(lc)
  return P3


def visible(P3, height, elev, azim, lo, hi, box=(4, 4, 3), steps=120):
  """Which lifted points the surface does not hide from an orthographic camera.

  Works in matplotlib's normalised box coordinates, marching from each point
  towards the camera and checking whether the surface rises above the ray.
  """
  lo, hi, box = np.asarray(lo), np.asarray(hi), np.asarray(box, dtype=float)
  e, a = np.radians(elev), np.radians(azim)
  view = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
  U = (P3 - lo) / (hi - lo) * box
  ok = np.ones(len(P3), dtype=bool)
  for s in np.linspace(0.02, 2.0 * box.max(), steps):
    Q = U + s * view
    inside = np.all((Q >= 0) & (Q <= box), axis=1) & ok
    if not inside.any():
      continue
    D = Q[inside] / box * (hi - lo) + lo
    ok[np.flatnonzero(inside)[height(D[:, :2]) > D[:, 2] + 0.01]] = False
  return ok


def arc_with_depth(ax, P3, lw, hidden_alpha=0.25, box=(4, 4, 3), **view):
  """The lifted arc, solid where visible and faint where the surface hides it."""
  lim = [ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()]
  lo, hi = np.array(lim)[:, 0], np.array(lim)[:, 1]
  vis = visible(P3, view["height"], view["elev"], view["azim"], lo, hi, box)
  seg = np.stack([P3[:-1], P3[1:]], axis=1)
  both = vis[:-1] & vis[1:]
  t = np.linspace(0, 1, len(seg))
  for mask, alpha, style in [(~both, hidden_alpha, (0, (2, 2))), (both, 1.0, "solid")]:
    if mask.any():
      lc = Line3DCollection(seg[mask], colors=TIME(t[mask]), linewidths=lw * (0.6 if alpha < 1 else 1),
                            alpha=alpha, linestyles=style, capstyle="round")
      ax.add_collection3d(lc)
  return vis


def contour_lines(ax, GX, GY, Z, levels, lift=0.002):
  """Level curves of the surface, drawn on it, so a path on it reads as crossing them."""
  cs = plt.figure().add_subplot().contour(GX, GY, Z, levels=levels)
  plt.close(cs.axes.figure)
  for level, segs in zip(cs.levels, cs.allsegs):
    for seg in segs:
      if len(seg) > 2:
        ax.plot(seg[:, 0], seg[:, 1], np.full(len(seg), level + lift),
                color=INK, lw=0.5, alpha=0.35)


def grey_relief(GX, GY, Z):
  """A plain grey surface, shaded by its slope, so colour is free for time."""
  from matplotlib.colors import LightSource
  shade = LightSource(azdeg=315, altdeg=45).hillshade(Z, dx=GX[0, 1] - GX[0, 0],
                                                        dy=GY[1, 0] - GY[0, 0], vert_exag=0.4)
  g = 0.72 + 0.22 * shade
  return np.dstack([g, g, g * 0.98, np.ones_like(g)])


def surface_axes(ax, positions, order):
  ax.set_zlim(0, 1)
  ax.set_zticks(positions); ax.set_zticklabels(order, fontsize=8)
  ax.set_xticks([]); ax.set_yticks([])
  ax.set_xlabel("PC1", labelpad=-10); ax.set_ylabel("PC2", labelpad=-10)
  ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False
  ax.view_init(elev=52, azim=-58)


def time_line(ax, P, lw=1.2, alpha=1.0):
  seg = np.stack([P[:-1], P[1:]], axis=1)
  lc = LineCollection(seg, cmap=TIME, linewidths=lw, alpha=alpha, capstyle="round")
  lc.set_array(np.linspace(0, 1, len(seg)))
  ax.add_collection(lc)


def ends(ax, P):
  ax.scatter(*P[0], s=70, facecolor="white", edgecolor=INK, lw=1.4, zorder=5)
  ax.scatter(*P[-1], s=70, marker="X", color=INK, zorder=5)


def box(P, margin):
  pad = margin * np.ptp(P, axis=0)
  return P.min(axis=0) - pad, P.max(axis=0) + pad


def plane(ax, X, margin=0.04):
  ax.set_aspect("equal")
  ax.set_xticks([]); ax.set_yticks([])
  for s in ax.spines.values():
    s.set_visible(False)
  lo, hi = box(X, margin)
  ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])


def time_bar(fig, ax):
  sm = plt.cm.ScalarMappable(cmap=TIME, norm=plt.Normalize(0, 1))
  cb = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02, ticks=[0, 1])
  cb.ax.set_yticklabels(["start", "end"])
  cb.outline.set_visible(False)


def time_axes(chapters, positions, order):
  """Reading position on x (chapters shaded and numbered), the mood scale on y."""
  fig, ax = plt.subplots(figsize=(8, 2.8))
  n = len(chapters)
  bounds = np.flatnonzero(np.diff(chapters)) + 0.5
  for k, (a, b) in enumerate(zip(np.r_[0, bounds], np.r_[bounds, n])):
    if k % 2:
      ax.axvspan(a, b, color="#f1f0ec", lw=0)
    ax.text((a + b) / 2, 1.0, str(k + 1), ha="center", va="bottom", fontsize=7,
            color="#8a8983", transform=ax.get_xaxis_transform())
  ax.set_xlim(0, n - 1)
  ax.set_yticks(positions); ax.set_yticklabels(order, fontsize=8)
  ax.set_xlabel("reading position (paragraph); chapters shaded")
  for s in ("top", "right"):
    ax.spines[s].set_visible(False)
  return fig, ax


def save(fig, name):
  path = os.path.join(HERE, name)
  fig.savefig(path)
  plt.close(fig)
  print(f"wrote {path}")


def main():
  X = mood_data.load_coords(BOOK, MODEL, "pca")
  _, emotions, scores, chapters, _ = mood_data.load_book(BOOK)
  m, order, positions, _ = mood(scores, emotions)

  # 1. the mess: every paragraph, joined in reading order
  fig, ax = plt.subplots(figsize=(6, 5))
  time_line(ax, X, lw=0.5, alpha=0.8)
  ax.scatter(*X.T, s=3, color=INK, alpha=0.35, lw=0)
  plane(ax, X); time_bar(fig, ax)
  save(fig, "1_paragraphs.png")

  # 2. windows: the same book at growing window sizes, zoomed on the arc
  sizes = [(1, 1), (10, 5), (20, 10), (40, 20)]
  focus = windows(X, 10, 5)
  fig, axes = plt.subplots(1, len(sizes), figsize=(4 * len(sizes), 3.6))
  for ax, (w, s) in zip(axes, sizes):
    ax.scatter(*X.T, s=2, color=FAINT, lw=0)
    P = windows(X, w, s)
    time_line(ax, P, lw=0.6 if w == 1 else 1.6)
    ends(ax, P)
    plane(ax, X if w == 1 else focus, 0.04 if w == 1 else 0.12)
    ax.set_title("single paragraphs" if w == 1 else f"windows of {w}, step {s}",
                 fontsize=10, color=INK)
  save(fig, "2_windows.png")

  # 3. the arc: the windows and the smooth curve through them
  P = windows(X, SIZE, STRIDE)
  # the project's arc: a uniform cubic B-spline fitted with bspline-regression
  fit = fit_curve(P, FitOpts(12, 3, 0.1, "dn", 100))
  curve = fit.curve
  fig, ax = plt.subplots(figsize=(6, 5))
  ax.scatter(*X.T, s=3, color=FAINT, lw=0)
  ax.plot(*P.T, color="#9a9993", lw=0.7, ls=":", zorder=2)
  ax.plot(*fit.control_points.T, color=INK, lw=0.8, ls="--", marker="s", ms=3.5,
          alpha=0.6, zorder=2)
  time_line(ax, curve, lw=3)
  ax.scatter(*P.T, c=np.linspace(0, 1, len(P)), cmap=TIME, s=28,
             edgecolor="white", lw=0.8, zorder=4)
  ends(ax, P)
  plane(ax, P, 0.15); time_bar(fig, ax)
  save(fig, "3_arc.png")



  # 5. the mood surface; 6. every paragraph lifted onto it (grey, so colour = time);
  # 7. the same surface with the arc lifted onto it
  GX, GY, Z, height_at = mood_surface.fit(X, m, h=0.2, resolution=260)
  base = MOOD(np.clip(Z, 0, 1))

  fig = plt.figure(figsize=(7, 5.6))
  ax = fig.add_subplot(111, projection="3d")
  ax.plot_surface(GX, GY, Z, facecolors=base, rstride=2, cstride=2, lw=0,
                  antialiased=True, shade=False)
  surface_axes(ax, positions, order)
  save(fig, "4_mood_surface.png")

  # the canonical route: the windows lifted and joined by exact geodesics on the
  # mood surface, each point timed by its shadow's length along the leg
  centers = np.array([(a + b - 1) / 2 for a, b in window_bounds(len(m), SIZE, STRIDE)])
  gx, gy, gz, _ = mood_surface.fit(X, m, h=0.2, resolution=140)
  runs, solved, total = mood_route(gx, gy, gz, P, centers)
  print(f"  route: {solved}/{total} geodesic legs")

  # 7. the geodesic route from three sides; parts a hill hides are drawn faint and dashed
  route = np.vstack([R for R, _ in runs])
  R3 = route + [0, 0, 0.004]
  W3 = np.column_stack([P, height_at(P) + 0.004])
  views = [(52, -58, "from above"), (22, -58, "from the side"), (30, 122, "from behind")]
  fig = plt.figure(figsize=(16, 5.4))
  for k, (elev, azim, title) in enumerate(views):
    ax = fig.add_subplot(1, 3, k + 1, projection="3d", computed_zorder=False)
    ax.set_proj_type("ortho"); ax.set_box_aspect((4, 4, 3))
    ax.plot_surface(GX, GY, Z, facecolors=base, rstride=2, cstride=2, lw=0,
                    antialiased=True, shade=False)
    surface_axes(ax, positions, order)
    ax.view_init(elev=elev, azim=azim)
    vis = arc_with_depth(ax, R3, 1.6, height=height_at, elev=elev, azim=azim)
    wvis = visible(W3, height_at, elev, azim, *np.array(
      [ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()]).T)
    ax.scatter(*W3[wvis].T, s=9, color="white", edgecolor=INK, lw=0.5, depthshade=False)
    for idx, kw in [(0, dict(facecolor="white", edgecolor=INK, lw=1.4)),
                    (-1, dict(marker="X", color=INK))]:
      ax.scatter(*W3[idx], s=60, depthshade=False, alpha=1.0 if wvis[idx] else 0.3, **kw)
    ax.set_title(title, fontsize=10, color=INK)
    print(f"  view {title}: {vis.mean():.0%} of the route visible")
  save(fig, "7_geodesic_route.png")

  # 8. close-up: the same surface, cropped to where the route runs
  lo, hi = box(route[:, :2], 0.25)
  cols = (GX[0] >= lo[0]) & (GX[0] <= hi[0])
  rows = (GY[:, 0] >= lo[1]) & (GY[:, 0] <= hi[1])
  CX, CY, CZ = GX[np.ix_(rows, cols)], GY[np.ix_(rows, cols)], Z[np.ix_(rows, cols)]
  fig = plt.figure(figsize=(9, 6.4))
  ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
  ax.set_proj_type("ortho"); ax.set_box_aspect((4, 4, 2))
  ax.plot_surface(CX, CY, CZ, facecolors=MOOD(np.clip(CZ, 0, 1)), rstride=1, cstride=1,
                  lw=0, antialiased=True, shade=False)
  zlo, zhi = CZ.min() - 0.02, CZ.max() + 0.02
  contour_lines(ax, CX, CY, CZ, np.linspace(CZ.min(), CZ.max(), 14)[1:-1])
  ax.set_zlim(zlo, zhi)
  keep = (positions >= zlo) & (positions <= zhi)
  ax.set_zticks(positions[keep]); ax.set_zticklabels(np.array(order)[keep], fontsize=9)
  ax.set_xticks([]); ax.set_yticks([])
  ax.set_xlabel("PC1", labelpad=-10); ax.set_ylabel("PC2", labelpad=-10)
  ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False
  ax.view_init(elev=38, azim=-58)
  vis = arc_with_depth(ax, R3, 1.8, height=height_at, elev=38, azim=-58, box=(4, 4, 2))
  wvis = visible(W3, height_at, 38, -58, *np.array(
    [ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()]).T, box=(4, 4, 2))
  ax.scatter(*W3[wvis].T, s=14, color="white", edgecolor=INK, lw=0.6, depthshade=False)
  ax.scatter(*W3[0], s=80, facecolor="white", edgecolor=INK, lw=1.5, depthshade=False)
  ax.scatter(*W3[-1], s=80, marker="X", color=INK, depthshade=False)
  print(f"  close-up: {vis.mean():.0%} of the route visible")
  save(fig, "8_route_closeup.png")

  # 8. what a geodesic does: two windows far apart (12 and 31), the shortest path
  # on the surface vs the straight line in the plane lifted onto it
  from geometry import mesh as gm
  from geometry.geodesic import exact_paths
  a, b = 12, 31
  Zj = gz + 1e-6 * np.random.default_rng(0).standard_normal(gz.shape)
  Vm, Fm, _ = gm.height_mesh(gx, gy, Zj, np.isfinite(Zj), 1.0)
  Vm, Fm, _ = gm.largest_component(Vm, Fm)
  feet, _ = gm.snap(P[[a, b]], Vm)
  geo = exact_paths(Vm, Fm, [(int(feet[0]), int(feet[1]))])[0]
  line = np.linspace(geo[0, :2], geo[-1, :2], 300)
  line3 = np.column_stack([line, height_at(line)])
  plen = lambda Q: np.r_[0, np.cumsum(np.linalg.norm(np.diff(Q[:, :2], axis=0), axis=1))]
  print(f"  windows {a}->{b}: climb straight {np.abs(np.diff(line3[:, 2])).sum():.2f}, "
        f"geodesic {np.abs(np.diff(geo[:, 2])).sum():.2f}")
  lo, hi = box(np.vstack([geo[:, :2], line]), 0.6)
  cols = (GX[0] >= lo[0]) & (GX[0] <= hi[0]); rows = (GY[:, 0] >= lo[1]) & (GY[:, 0] <= hi[1])
  EX, EY, EZ = GX[np.ix_(rows, cols)], GY[np.ix_(rows, cols)], Z[np.ix_(rows, cols)]
  PINK, fig = "#e0457b", plt.figure(figsize=(16, 5.2))
  ax = fig.add_subplot(1, 3, 1, projection="3d", computed_zorder=False)
  ax.set_proj_type("ortho"); ax.set_box_aspect((4, 4, 2.4))
  ax.plot_surface(EX, EY, EZ, facecolors=MOOD(np.clip(EZ, 0, 1)), rstride=1, cstride=1,
                  lw=0, antialiased=True, shade=False)
  contour_lines(ax, EX, EY, EZ, np.linspace(EZ.min(), EZ.max(), 12)[1:-1])
  ax.plot(*line3.T + np.array([[0], [0], [0.004]]), color=INK, lw=1.2, ls="--")
  ax.plot(geo[:, 0], geo[:, 1], geo[:, 2] + 0.004, color=PINK, lw=1.8)
  ax.set_zlim(EZ.min() - 0.02, EZ.max() + 0.02); ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
  ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False
  ax.view_init(elev=35, azim=-58)
  ax.set_title("on the surface", fontsize=10, color=INK)
  ax = fig.add_subplot(1, 3, 2)
  ax.contourf(EX, EY, EZ, levels=16, cmap=MOOD, vmin=0, vmax=1)
  ax.contour(EX, EY, EZ, levels=16, colors=INK, linewidths=0.4, alpha=0.4)
  ax.plot(*line.T, color=INK, lw=1.2, ls="--", label="straight in the plane, lifted")
  ax.plot(geo[:, 0], geo[:, 1], color=PINK, lw=1.8, label="geodesic on the surface")
  for k, q in [(a, geo[0]), (b, geo[-1])]:
    ax.scatter(*q[:2], s=50, color="white", edgecolor=INK, zorder=5)
    ax.annotate(f"window {k}", q[:2], xytext=(6, 6), textcoords="offset points", fontsize=9)
  ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
  ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
  ax.set_title("from above, over the mood map", fontsize=10, color=INK)
  ax = fig.add_subplot(1, 3, 3)
  ax.plot(plen(line3) / plen(line3)[-1], line3[:, 2], color=INK, lw=2, ls="--")
  ax.plot(plen(geo) / plen(geo)[-1], geo[:, 2], color=PINK, lw=3)
  ax.set_yticks(positions); ax.set_yticklabels(order, fontsize=8)
  ax.set_ylim(min(line3[:, 2].min(), geo[:, 2].min()) - 0.03, max(line3[:, 2].max(), geo[:, 2].max()) + 0.03)
  ax.set_xlabel("fraction of the way"); ax.set_title("mood along each path", fontsize=10, color=INK)
  for sp in ("top", "right"): ax.spines[sp].set_visible(False)
  save(fig, "6_geodesic_vs_straight.png")

  grey = grey_relief(GX, GY, Z)
  fig = plt.figure(figsize=(13, 5.6))
  for k, title in enumerate(["every paragraph, joined in reading order",
                             "every paragraph as a dot"]):
    ax = fig.add_subplot(1, 2, k + 1, projection="3d", computed_zorder=False)
    ax.plot_surface(GX, GY, Z, facecolors=grey, rstride=2, cstride=2, lw=0,
                    antialiased=True, shade=False)
    if k == 0:
      draped(ax, X, 0.6, height_at, alpha=0.8)
    else:
      ax.scatter(X[:, 0], X[:, 1], height_at(X) + 0.004, c=np.linspace(0, 1, len(X)),
                 cmap=TIME, s=9, lw=0, depthshade=False)
    surface_axes(ax, positions, order)
    ax.set_title(title, fontsize=10, color=INK)
  save(fig, "5_paragraphs_lifted.png")

  # 8. every paragraph lifted, over reading time, with the route through them;
  # 9. the route alone, zoomed to the band it moves in.
  arc_t = np.vstack([np.column_stack([t, R[:, 2]]) for R, t in runs])
  lifted = arc_t


  fig, ax = time_axes(chapters, positions, order)
  time_line(ax, arc_t, lw=2.4)
  below = positions[positions <= arc_t[:, 1].min()].max(initial=0.0)
  above = positions[positions >= arc_t[:, 1].max()].min(initial=1.0)
  ax.set_ylim(below - 0.02, above + 0.02)
  save(fig, "10_mood_over_time.png")




if __name__ == "__main__":
  main()
