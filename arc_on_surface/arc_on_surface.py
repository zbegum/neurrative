"""
The narrative arc as a curve that actually lies on the emotion surface.

The goal: take the surface already fitted in `surface/`, take the 2-D narrative
arc, and draw the one curve on that surface whose **vertical shadow is the
narrative arc**. Then measure it with the same geodesic machinery the rest of the
geometry uses, so "how expensive is the story's route" becomes a number.

    C(t) = ( arc_x(t), arc_y(t), Surface( arc_x(t), arc_y(t) ) )

That is the unique lift of the 2-D arc onto the surface: project C back down and
you get the narrative arc exactly, by construction. Only PCA supports it -- the
plane has to be metric for a point to have one well-defined height.

Why this file was rewritten
---------------------------
The previous version sampled the surface only at the ~157 window centres and
drew straight 3-D chords between them. Those chords do not lie on the surface:
measured against the terrain, 92 of 156 segments were off by more than 0.01 and
the worst by 0.14 -- 30% of the total relief -- because each chord spanned about
ten grid cells of a curved surface. It also never touched the mesh or the
geodesic solver, so nothing connected it to the rest of the pipeline.

The fix is to resample the 2-D arc by arc length until consecutive samples are
within a fraction of a grid cell, then lift every sample. `--oversample` sets
that fraction; the run prints the residual so the claim is checked, not asserted.

What gets measured
------------------
The lifted curve is measured on the mesh `geometry/mesh.py` builds from this
surface -- the same terrain `geodesic_arrows.py` walks:

  projected length   the arc's length in the flat PCA plane, ignoring emotion.
  surface length     its length in R^3 once the terrain is included.
  climb ratio        surface / projected. What the relief cost the route.
  excess             surface / summed geodesic length **between consecutive
                     waypoints**. This is the one that answers "does the story
                     climb over emotional peaks, or take the cheap path around
                     them?" -- 1.0 means every step happened to follow the
                     geodesic, above 1 means it climbed more than it had to.

All of it is measured only where the surface is supported. The density floor
cuts holes in the terrain and the arc runs through them; the kernel still
returns a height there, but from too few paragraphs to mean anything. So the
route is treated as several disconnected segments, `surface_length` is the sum
of the supported ones rather than a single number for the book, and
`supported_fraction` / `n_segments` record what that leaves out.

Note that comparing the *whole* arc against the geodesic between its endpoints
is a different and largely meaningless question here: the arc wanders and ends
near where it began, so that ratio came out at 173x and measured how much the
story moved, not how expensively. It is recorded as context, not as a finding.

`--alpha` is vertical exaggeration and it decides all of this: it is the exchange
rate between score units and PCA units, and nothing in the data fixes it. At
alpha = 0 the surface is flat and every ratio collapses to 1.

Example:

  python arc_on_surface/arc_on_surface.py --book alice_wonderland --model bge-m3
  python arc_on_surface/arc_on_surface.py --model bge-m3 --alpha 2 --emotions wonder
"""

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np

# The repository root and common/ on the path, wherever this script lives.
_ROOT = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_ROOT, "common")):
  _ROOT = os.path.dirname(_ROOT)
sys.path[1:1] = [_ROOT, os.path.join(_ROOT, "common")]

import paths
from data import load_paragraphs, load_scores
# The windowing is shared: `windows.py` is the one definition of what a window
# is, and what saves the series these scripts draw. Re-exported because five
# other arc scripts import `window_bounds` from this module.
from windows import window_bounds
from smooth_common import Standardizer, load_pca, fit_grid
from geometry.smoothers import gaussian_nw
from geometry import mesh as gmesh
from geometry.geodesic import edge_flip_paths, polyline_length

CMAP = "magma"
UNSUPPORTED = "0.55"


def window_points(X_raw, windows, how="mean"):
  """The (x, y) that stands for each window, and the paragraph behind it.

  Returns `(points, index)`, where `index` is the paragraph each point is -- or
  None for `mean`, which is the point of the distinction:

    mean    the centroid of the window's paragraphs. No paragraph is there. The
            arc passes through positions the book never occupies, and because a
            centroid is pulled toward the middle of whatever it averages, the
            whole arc shrinks inward: at w10 s5 it covers 30% of the cloud's
            bounding area, at w40 s20 only 13%. 15% of these centroids land
            below the surface's density floor -- in the gaps between clusters,
            where there is no terrain to stand on.

    medoid  the paragraph in the window closest to all the others in it. A real
            paragraph, so it is always somewhere the book actually goes: only
            3% fall below the floor, the route stops floating over holes, and
            every waypoint can be opened and read. One point per window,
            shared by all six emotions, so it is still a single arc.

  A window's paragraphs are spread far apart in this space, so the medoid stands
  in for a group that is not really in one place -- but it stands in for it as a
  place in the book rather than as an average of coordinates.
  """
  if how == "mean":
    return np.array([X_raw[s:e].mean(axis=0) for s, e in windows]), None
  if how != "medoid":
    raise ValueError(f"Unknown window point {how!r}; expected mean or medoid.")

  idx = []
  for s, e in windows:
    P = X_raw[s:e]
    D = np.linalg.norm(P[:, None] - P[None], axis=2)
    idx.append(s + int(D.sum(axis=1).argmin()))
  idx = np.asarray(idx)
  return X_raw[idx], idx


def geodesic_route(GX, GY, Z, waypoints, alpha, snap_tol, t_of_waypoint=None):
  """The route as geodesics between consecutive waypoints, not straight chords.

  A straight leg is a line in PCA coordinates: both ends can be real paragraphs
  while the segment between them crosses a gap where no paragraph lies and the
  surface has been masked away. That is where the extrapolation comes from -- not
  from the waypoints. 26 of 156 legs do it on Alice.

  A geodesic leg cannot: it is computed on the mesh, and the mesh only exists
  where the surface is supported. So the route is confined to real terrain by
  construction, and it lies on the surface exactly rather than to within a chord
  error. What it gives up is the lift property -- the shadow of this curve is no
  longer the straight-line 2-D arc, because the path bends around the holes.

  Returns `(runs, stats)`. Each run is `(P, tt)`: an (n, 3) polyline in
  (x, y, score) units and the reading position of each of its points. A dropped
  leg -- an endpoint off the mesh, or two endpoints in different components --
  ends the current run and starts the next, so the route is honestly broken
  rather than bridged by a line that was never solved.
  """
  vertices, faces, _ = gmesh.height_mesh(GX, GY, Z, np.isfinite(Z), alpha)
  vertices, faces, _ = gmesh.largest_component(vertices, faces)
  graph = gmesh.edge_graph(vertices, faces)

  feet, dist = gmesh.snap(waypoints, vertices)
  ok = dist <= snap_tol
  pairs_idx = [i for i in range(len(feet) - 1) if ok[i] and ok[i + 1]]
  solved = edge_flip_paths(vertices, faces,
                           [(int(feet[i]), int(feet[i + 1])) for i in pairs_idx],
                           graph=graph)
  legs = dict(zip(pairs_idx, solved))

  if t_of_waypoint is None:
    t_of_waypoint = np.arange(len(waypoints), dtype=float)

  runs, cur_P, cur_t = [], [], []
  for i in range(len(feet) - 1):
    leg = legs.get(i)
    if leg is None or alpha <= 0:
      if cur_P:
        P, tt = np.vstack(cur_P), np.concatenate(cur_t)
        if len(P) > 1:
          runs.append((P, tt))
      cur_P, cur_t = [], []
      continue
    P = np.column_stack([leg[:, 0], leg[:, 1], leg[:, 2] / alpha])
    # Reading position along the leg, by arc length, so colour still means time.
    seg = np.linalg.norm(np.diff(P[:, :2], axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    frac = s / s[-1] if s[-1] > 0 else np.zeros(len(P))
    tt = t_of_waypoint[i] + frac * (t_of_waypoint[i + 1] - t_of_waypoint[i])
    if cur_P:  # drop the shared endpoint
      P, tt = P[1:], tt[1:]
    cur_P.append(P); cur_t.append(tt)
  if cur_P:
    P, tt = np.vstack(cur_P), np.concatenate(cur_t)
    if len(P) > 1:
      runs.append((P, tt))

  stats = {"legs_total": len(feet) - 1,
           "legs_solved": sum(1 for i in range(len(feet) - 1)
                              if legs.get(i) is not None),
           "runs": len(runs),
           "waypoints_on_mesh": int(ok.sum())}
  return runs, stats


def build_surface(X_raw, y_raw, hx, hy, resolution, margin, density_floor):
  """The emotion terrain over the PCA plane -- same protocol as geodesic_arrows.

  Fit in standardized units (so h means the same across emotions), return the
  grid in raw PCA coordinates with the low-density cells masked to NaN.
  """
  xs, ys = Standardizer().fit(X_raw), Standardizer().fit(y_raw)
  gx, gy, Zz, mask, _ = fit_grid(
    gaussian_nw, {"hx": hx, "hy": hy},
    xs.forward(X_raw), ys.forward(y_raw), resolution, margin, density_floor)
  Z = ys.inverse(Zz)
  raw = xs.inverse(np.column_stack([gx.ravel(), gy.ravel()]))
  return (raw[:, 0].reshape(gx.shape), raw[:, 1].reshape(gy.shape), Z, xs, ys)


def surface_at(xs, ys, X_raw, y_raw, Q, hx, hy):
  """Evaluate the emotion surface at arbitrary plane points Q (the arc)."""
  vals, _ = gaussian_nw(xs.forward(X_raw), ys.forward(y_raw),
                        xs.forward(Q), hx=hx, hy=hy)
  return np.clip(ys.inverse(vals), 0.0, 1.0)


def support_at(xs, ys, X_raw, y_raw, Q, hx, hy, floor_pct):
  """Kernel support at Q, and the floor the drawn surface is masked at.

  `surface_at` returns a height everywhere, because a Gaussian kernel has
  infinite support -- including over the holes the density floor cuts out of the
  surface. Asking for the support separately is what lets the route be measured
  only where the terrain exists, instead of silently extrapolating across it.
  """
  _, at_samples = gaussian_nw(xs.forward(X_raw), ys.forward(y_raw),
                              xs.forward(X_raw), hx=hx, hy=hy)
  # Negative means no floor at all, matching fit_grid: everything is supported
  # because nothing was masked away.
  floor = (-np.inf if floor_pct < 0
           else float(np.percentile(at_samples, floor_pct)))
  _, sup = gaussian_nw(xs.forward(X_raw), ys.forward(y_raw),
                       xs.forward(Q), hx=hx, hy=hy)
  return sup, floor


def _masked_length(P, keep):
  """Length of a polyline, counting only the segments `keep` selects."""
  if len(P) < 2:
    return 0.0
  return float(np.linalg.norm(np.diff(P, axis=0), axis=1)[keep].sum())


def runs(good):
  """Contiguous [start, stop) index runs of True in a boolean array."""
  out, i = [], 0
  while i < len(good):
    if good[i]:
      j = i
      while j < len(good) and good[j]:
        j += 1
      out.append((i, j))
      i = j
    else:
      i += 1
  return out


def resample(path_xy, spacing):
  """Resample a 2-D polyline by arc length, at most `spacing` between samples.

  Returns (points, t) where t is each sample's position along the original
  polyline in *index* units, so per-window quantities can be interpolated onto
  the dense curve without a second parameterization.

  This is the step that makes the lift lie on the surface: a chord is only as
  close to a curved terrain as it is short, so the sampling has to be fine
  relative to the grid, not to the windows.
  """
  seg = np.linalg.norm(np.diff(path_xy, axis=0), axis=1)
  s = np.concatenate([[0.0], np.cumsum(seg)])
  n = max(int(np.ceil(s[-1] / spacing)) + 1, len(path_xy))
  su = np.linspace(0.0, s[-1], n)
  x = np.interp(su, s, path_xy[:, 0])
  y = np.interp(su, s, path_xy[:, 1])
  t = np.interp(su, s, np.arange(len(path_xy), dtype=float))
  return np.column_stack([x, y]), t


def lies_on_surface(curve_xy, z, xs, ys, X_raw, y_raw, hx, hy):
  """Largest gap between a chord's midpoint and the surface beneath it.

  The honest test of "this curve is on the surface": if the polyline is fine
  enough, the straight segment between consecutive samples never departs from
  the terrain by more than interpolation error.
  """
  if len(curve_xy) < 2:
    return 0.0
  mid_xy = (curve_xy[:-1] + curve_xy[1:]) / 2.0
  z_chord = (z[:-1] + z[1:]) / 2.0
  z_true = surface_at(xs, ys, X_raw, y_raw, mid_xy, hx, hy)
  return float(np.nanmax(np.abs(z_chord - z_true)))


def route_metrics(curve_xy, z, t, waypoints, GX, GY, Z, alpha, snap_tol,
                  supported):
  """What the terrain cost the story's route, against the cheapest alternative.

  Everything here is measured **only where the surface is supported**. The
  density floor cuts holes in the terrain, and the arc passes through them; a
  Gaussian kernel still returns a height there, but it is extrapolated from
  under two paragraphs of evidence and there is no drawn surface beneath it.
  Including those stretches would let unmeasured ground into every ratio.

  The consequence is that the route is a set of disconnected segments rather
  than one curve, and `surface_length` is the summed length of the supported
  parts, not a single number for the whole book. `supported_fraction` and
  `n_segments` say how much of the story that leaves out.

  Two comparisons, and only the second answers the question the folder README
  poses ("does the story climb over emotional peaks, or take the cheap path
  around them?"):

    climb ratio   surface length / projected length. How much longer the route
                  is once the terrain is included. Pure cost of relief.
    excess        surface length / summed geodesic length **between consecutive
                  waypoints**. The story's actual route from one window to the
                  next, against the shortest route on the same surface between
                  the same two points. Above 1 means it climbed more than it had
                  to; 1 means it happened to follow the geodesic.

  Comparing the whole arc against the geodesic between its *endpoints* is not
  the same question and is close to meaningless here: the arc wanders and comes
  back near where it started, so that ratio measures how much the story moved,
  not how expensively it moved. It is reported as context only.
  """
  out = {"alpha": float(alpha)}
  z = np.asarray(z, dtype=float)
  supported = np.asarray(supported, dtype=bool)

  flat = np.column_stack([curve_xy, np.zeros(len(curve_xy))])
  lifted = np.column_stack([curve_xy, alpha * z])

  # A segment counts only if both of its ends stand on supported terrain.
  keep = supported[:-1] & supported[1:]
  out["supported_fraction"] = float(supported.mean())
  out["n_segments"] = len(runs(keep))
  out["projected_length"] = _masked_length(flat, keep)
  out["surface_length"] = _masked_length(lifted, keep)
  out["climb_ratio"] = (out["surface_length"] / out["projected_length"]
                        if out["projected_length"] > 0 else None)

  dz = np.diff(alpha * z)[keep]
  out["total_ascent"] = float(dz[dz > 0].sum())
  out["total_descent"] = float(-dz[dz < 0].sum())

  vertices, faces, _ = gmesh.height_mesh(GX, GY, Z, np.isfinite(Z), alpha)
  # The exact solver asserts on a disconnected mesh, and the density mask does
  # fragment this field into islands at these bandwidths.
  vertices, faces, kept = gmesh.largest_component(vertices, faces)
  out["mesh_vertices"], out["mesh_faces"] = int(len(vertices)), int(len(faces))
  out["mesh_kept_fraction"] = kept
  graph = gmesh.edge_graph(vertices, faces)

  feet, snap_dist = gmesh.snap(waypoints, vertices)
  out["waypoint_snap_max"] = float(snap_dist.max())

  # A waypoint that fell outside the masked region snaps to whatever vertex
  # survived nearest, which here can be a whole leg away (snap 0.065 against a
  # median leg of 0.067). Its geodesic would then start somewhere the route
  # never goes, and the comparison stops being like-for-like -- that is what
  # produced a sadness ratio of 0.97, a route apparently shorter than the
  # shortest path. Those legs are dropped and counted, not silently averaged in.
  on_surface = snap_dist <= snap_tol
  out["waypoints_on_surface"] = int(on_surface.sum())
  out["waypoints_total"] = int(len(feet))

  # Waypoint i -> i+1, the story's own steps, where both feet are trustworthy.
  idx = [i for i in range(len(feet) - 1) if on_surface[i] and on_surface[i + 1]]
  pairs = [(int(feet[i]), int(feet[i + 1])) for i in idx]
  legs = edge_flip_paths(vertices, faces, pairs, graph=graph)

  geo_total, route_total, ratios, solved, unsupported = 0.0, 0.0, [], 0, 0
  for i, leg in zip(idx, legs):
    if leg is None:
      continue
    # The story's own route over the same interval: the lifted dense curve
    # between waypoint i and i+1.
    span = (t >= i) & (t <= i + 1)
    seg = lifted[span]
    if len(seg) < 2:
      continue
    # A leg whose route crosses a hole has no route length to compare: part of
    # it is extrapolation. The geodesic exists (it runs on the meshed part), but
    # the ratio would be route-with-a-guess over shortest-path, so drop it.
    if not supported[span].all():
      unsupported += 1
      continue
    g, r = float(polyline_length(leg)), float(polyline_length(seg))
    if g <= 0:
      continue
    geo_total += g
    route_total += r
    ratios.append(r / g)
    solved += 1

  out["legs_solved"] = solved
  out["legs_dropped_unsupported"] = unsupported
  out["legs_eligible"] = len(pairs)
  out["legs_total"] = int(len(feet) - 1)
  out["geodesic_leg_length"] = geo_total or None
  out["excess_over_geodesic"] = (route_total / geo_total) if geo_total else None
  out["leg_ratio_median"] = float(np.median(ratios)) if ratios else None
  out["leg_ratio_p90"] = float(np.percentile(ratios, 90)) if ratios else None

  # Context only -- see the docstring.
  ends = (edge_flip_paths(vertices, faces, [(int(feet[0]), int(feet[-1]))],
                          graph=graph)[0]
          if on_surface[0] and on_surface[-1] else None)
  out["endpoint_geodesic_length"] = (float(polyline_length(ends))
                                     if ends is not None else None)
  out["geodesic_path"] = ends
  return out


def draw_on_surface(ax, GX, GY, Z, curve_xy, z, emotion, geodesic=None, alpha=1.0,
                    supported=None):
  """Terrain with the lifted arc on it, and optionally the geodesic beneath.

  The route is drawn split the same way it is measured: coloured where the
  surface is supported, grey where it is extrapolated across a hole.
  """
  ax.plot_surface(GX, GY, Z, cmap=CMAP, vmin=0.0, vmax=1.0, linewidth=0,
                  antialiased=True, alpha=0.5, rstride=3, cstride=3)

  pts = np.column_stack([curve_xy[:, 0], curve_xy[:, 1], z])
  seg = np.stack([pts[:-1], pts[1:]], axis=1)
  keep = (np.ones(len(seg), dtype=bool) if supported is None
          else (supported[:-1] & supported[1:]))
  order = np.linspace(0, 1, len(seg))
  if keep.any():
    lc = Line3DCollection(seg[keep], cmap="plasma", linewidth=2.4, zorder=5)
    lc.set_array(order[keep])
    lc.set_clim(0.0, 1.0)
    ax.add_collection3d(lc)
  if (~keep).any():
    ax.add_collection3d(Line3DCollection(seg[~keep], colors=UNSUPPORTED,
                                         linewidths=1.2, zorder=4))

  # The cheapest route between the same endpoints, drawn back in score units so
  # it shares the arc's axis rather than the mesh's exaggerated one.
  if geodesic is not None and alpha > 0:
    ax.plot(geodesic[:, 0], geodesic[:, 1], geodesic[:, 2] / alpha,
            color="#1baf7a", lw=1.6, ls="--", zorder=6)

  ax.scatter(*pts[0], color="black", s=45, marker="o", depthshade=False, zorder=7)
  ax.scatter(*pts[-1], color="black", s=55, marker="X", depthshade=False, zorder=7)
  ax.set_zlim(0.0, 1.0)
  ax.set_xticklabels([]); ax.set_yticklabels([])
  ax.set_title(emotion, fontsize=11, pad=0)
  ax.set_zlabel(emotion, fontsize=8)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--book", default="alice_wonderland")
  ap.add_argument("--model", default="bge-m3")
  ap.add_argument("--size", default=10, type=int)
  ap.add_argument("--stride", default=5, type=int)
  ap.add_argument("--emotions", nargs="*", default=None,
                  help="Subset of emotions (default: all).")
  # 0.15 (not the geodesic 0.2): below the CV-tuned 0.4, in the bandwidth grid's
  # "noise -> structure" band, so the terrain has visible relief and is not flat.
  ap.add_argument("--hx", default=0.15, type=float)
  ap.add_argument("--hy", default=0.15, type=float)
  ap.add_argument("--alpha", default=1.0, type=float,
                  help="Vertical exaggeration: the exchange rate between score "
                       "units and PCA units. Sets every length ratio below.")
  ap.add_argument("--snap-tol", default=1.0, type=float,
                  help="How far (in grid cells) a waypoint may sit from the "
                       "meshed surface and still be trusted as a geodesic "
                       "endpoint. Legs failing this are dropped and reported.")
  ap.add_argument("--oversample", default=4.0, type=float,
                  help="Curve samples per grid cell. Higher = the lift hugs the "
                       "terrain more tightly; the run reports the residual.")
  ap.add_argument("--legs", default="straight", choices=["straight", "geodesic"],
                  help="How consecutive waypoints are joined. 'straight' is a "
                       "line in the PCA plane, lifted onto the surface -- the "
                       "curve this file was written around, whose shadow is the "
                       "2-D arc. 'geodesic' is the shortest path along the "
                       "surface instead: it cannot leave the terrain and lies "
                       "on it exactly, but it is per-emotion and its shadow is "
                       "no longer the 2-D arc.")
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

  if args.emotions:
    unknown = [e for e in args.emotions if e not in emotions]
    if unknown:
      raise ValueError(f"Unknown emotion(s) {unknown}. Available: {emotions}")
    selected = args.emotions
  else:
    selected = emotions

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
  # The arc: moving average of the PCA coordinates in reading order. Because the
  # plane is the surface's own domain, this needs no reprojection.
  arc_xy, arc_idx = window_points(X_raw, windows, args.window_point)

  print(f"=== the arc as a curve on the surface: {args.book} / {args.model} ===")
  print(f"  {n} paragraphs -> {len(windows)} windows "
        f"(size {args.size} stride {args.stride}), alpha {args.alpha}")

  # One grid, one sampling rate, shared by every emotion so the curves are
  # comparable. The grid step is set by the surface, not by the windows.
  probe = build_surface(X_raw, matrix[:, 0], args.hx, args.hy,
                        args.resolution, args.margin, args.density_floor)
  cell = float(abs(probe[0][0, 1] - probe[0][0, 0]))
  dense_xy, t = resample(arc_xy, cell / args.oversample)
  print(f"  grid cell {cell:.4f} -> resampled {len(arc_xy)} -> {len(dense_xy)} "
        f"points ({args.oversample:g} per cell)\n")

  terrains, heights, own, paired, metrics = {}, {}, {}, {}, []
  supported, curves = {}, {}
  for e in selected:
    y_raw = matrix[:, emotions.index(e)]
    GX, GY, Z, xs, ys = build_surface(X_raw, y_raw, args.hx, args.hy,
                                      args.resolution, args.margin,
                                      args.density_floor)

    if args.legs == "geodesic":
      # The route is the geodesics themselves, so it is per-emotion: the
      # cheapest way across the wonder terrain is not the way across danger's.
      # Every point is on the mesh, so nothing is unsupported and there is no
      # chord error -- the two things the straight route has to be measured for.
      runs, st = geodesic_route(GX, GY, Z, arc_xy, args.alpha,
                                args.snap_tol * cell)
      P = np.vstack([r[0] for r in runs])
      curve_xy, z = P[:, :2], P[:, 2]
      t_e = np.concatenate([r[1] for r in runs])
      supported[e] = np.ones(len(P), dtype=bool)
      legs_note = f"{st['legs_solved']}/{st['legs_total']} legs, {st['runs']} run(s)"
    else:
      curve_xy, t_e = dense_xy, t
      z = surface_at(xs, ys, X_raw, y_raw, dense_xy, args.hx, args.hy)
      sup, floor = support_at(xs, ys, X_raw, y_raw, dense_xy, args.hx, args.hy,
                              args.density_floor)
      supported[e] = sup >= floor
      legs_note = "straight"

    curves[e] = (curve_xy, t_e)
    heights[e] = z

    # The paired comparison lives here, at the windows themselves -- one surface
    # height and one felt emotion per window, both real, nothing interpolated.
    # The dense curve carries 6500 points but only 157 of them are windows, so
    # averaging a residual over it would be averaging mostly filler.
    z_win = surface_at(xs, ys, X_raw, y_raw, arc_xy, args.hx, args.hy)
    own_win = np.array([y_raw[s:e2].mean() for s, e2 in windows])
    paired[e] = (z_win, own_win)
    # ...and the same pair spread along the dense curve, for the profile plot.
    own[e] = np.interp(t_e, np.arange(len(windows)), own_win)

    residual = lies_on_surface(curve_xy, z, xs, ys, X_raw, y_raw, args.hx, args.hy)
    m = route_metrics(curve_xy, z, t_e, arc_xy, GX, GY, Z, args.alpha,
                      args.snap_tol * cell, supported[e])
    geo = m.pop("geodesic_path")
    if args.legs == "geodesic":
      # The route IS the geodesic, so this ratio is 1.0 by construction and says
      # nothing. What is worth recording instead is how much longer the terrain-
      # following route is than the straight one it replaces.
      m["excess_over_geodesic"] = None
      m["leg_ratio_median"] = m["leg_ratio_p90"] = None
      straight = np.column_stack([
        dense_xy, args.alpha * surface_at(xs, ys, X_raw, y_raw, dense_xy,
                                          args.hx, args.hy)])
      m["straight_surface_length"] = float(polyline_length(straight))
      m["detour_over_straight"] = (m["surface_length"] /
                                   m["straight_surface_length"]
                                   if m["straight_surface_length"] else None)
    gap = np.abs(z_win - own_win)
    m.update(emotion=e, legs=args.legs, window_point=args.window_point,
             off_surface_max=residual,
             n_windows=int(len(arc_xy)), n_curve_points=int(len(curve_xy)),
             mean_abs_surface_minus_own=float(gap.mean()),
             max_abs_surface_minus_own=float(gap.max()),
             surface_above_own=int((z_win > own_win).sum()),
             height_min=float(z.min()), height_max=float(z.max()))
    metrics.append(m)
    terrains[e] = (GX, GY, Z, geo)

    exc = f"{m['excess_over_geodesic']:.3f}" if m["excess_over_geodesic"] else "n/a"
    print(f"  {e:>10}: height {z.min():.2f}..{z.max():.2f}  "
          f"off-surface {residual:.5f}  climb {m['climb_ratio']:.3f}  "
          f"gap@windows {m['mean_abs_surface_minus_own']:.3f}  "
          f"excess over geodesic {exc}  "
          f"({m['legs_solved']}/{m['legs_total']} legs, "
          f"{m['waypoints_on_surface']}/{m['waypoints_total']} waypoints on-surface)")
    print(f"  {'':>10}  measured on {100 * m['supported_fraction']:.0f}% of the "
          f"route, in {m['n_segments']} supported segment(s); "
          f"{m['legs_dropped_unsupported']} leg(s) dropped for crossing a hole")

  with open(os.path.join(out_dir, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

  # The 157 real points, so the pairing is usable and not just drawn: window
  # centre, its surface height, and the emotion it actually carried.
  np.savez(os.path.join(out_dir, paths.named("windows", "heights", "npz")),
           arc_xy=arc_xy, window_centers=centers,
           dense_xy=dense_xy, dense_t=t,
           **{f"surface_{e}": paired[e][0] for e in selected},
           **{f"own_{e}": paired[e][1] for e in selected})

  paths.stamp(out_dir, __file__, args, stack="geometry.smoothers",
              estimator="gaussian_nw (anisotropic hx, hy; standardized coords)",
              curve="lift of the 2-D arc: z = Surface(arc_x, arc_y)",
              geodesic="geometry.geodesic.edge_flip_paths (exact MMP)",
              n_curve_points=int(len(dense_xy)), emotions=selected)

  # --- figure 1: the lifted arc on each terrain, with its geodesic -------------
  cols = min(3, len(selected))
  rows = int(np.ceil(len(selected) / cols))
  fig = plt.figure(figsize=(6 * cols, 5.3 * rows))
  for i, e in enumerate(selected):
    ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
    GX, GY, Z, geo = terrains[e]
    draw_on_surface(ax, GX, GY, Z, curves[e][0], heights[e], e, geo, args.alpha,
                    supported[e])
  if args.legs == "geodesic":
    sub = (f"coloured = geodesic legs between {args.window_point} waypoints: on "
           f"the surface everywhere, but its shadow is not the 2-D arc\n"
           f"dashed green = the endpoint geodesic, context only")
  else:
    sub = (f"coloured = the story's route, whose shadow is the 2-D narrative arc; "
           f"grey = unsupported terrain, not measured; "
           f"dashed green = the endpoint geodesic, context only")
  fig.suptitle(
    f"The narrative arc as a curve on the emotion surface -- {args.book} / {args.model}\n"
    f"{sub} (alpha {args.alpha})", fontsize=13)
  fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.02,
                      wspace=0.05, hspace=0.12)
  p = os.path.join(out_dir, paths.named("arc_on_surface", "3d"))
  fig.savefig(p, dpi=170); plt.close(fig); print(f"\n  wrote {p}")

  # --- figure 2: height along the route, against what the text carried ---------
  fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 3.0 * rows),
                           sharex=True, squeeze=False)
  for ax, e in zip(axes.ravel(), selected):
    para = np.interp(curves[e][1], np.arange(len(windows)), centers)
    # Break the height where the surface is unsupported, so the profile shows
    # the same gaps the metrics are measured over rather than a smooth guess.
    ok = supported[e]
    ax.plot(para, np.where(ok, heights[e], np.nan), color="#d1495b", lw=1.8,
            label="surface height (from location)")
    ax.plot(para, np.where(ok, np.nan, heights[e]), color=UNSUPPORTED, lw=1.0,
            label="extrapolated (unsupported)")
    ax.plot(para, own[e], color="0.45", lw=1.2, ls="--", label="window's own emotion")
    ax.fill_between(para, heights[e], own[e], where=ok, color="#d1495b", alpha=0.12)
    ax.set_title(e, fontsize=11); ax.set_ylim(0, 1); ax.grid(alpha=0.25)
  for ax in axes[-1]:
    ax.set_xlabel("reading position (paragraph)")
  for ax in axes[:, 0]:
    ax.set_ylabel("emotion score")
  axes[0, 0].legend(loc="upper right", fontsize=8, framealpha=0.9)
  fig.suptitle("Height along the route: what the location predicts (solid) vs "
               "what the text carried (dashed)", fontsize=12)
  fig.tight_layout(rect=(0, 0, 1, 0.94))
  p = os.path.join(out_dir, paths.named("arc_on_surface", "heights"))
  fig.savefig(p, dpi=180); plt.close(fig); print(f"  wrote {p}")

  # --- figure 3: the two heights at the windows, paired -----------------------
  # Same 157 (x, y). One curve sits on the terrain, the other at the emotion the
  # text actually carried, and the stems between them are the disagreement. This
  # is the only figure where every plotted point is a real window.
  fig = plt.figure(figsize=(6 * cols, 5.3 * rows))
  for i, e in enumerate(selected):
    ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
    GX, GY, Z, _ = terrains[e]
    z_win, own_win = paired[e]
    ax.plot_surface(GX, GY, Z, cmap=CMAP, vmin=0.0, vmax=1.0, linewidth=0,
                    antialiased=True, alpha=0.35, rstride=3, cstride=3)

    # The gap, drawn first so both curves sit on top of it.
    for j in range(len(arc_xy)):
      ax.plot([arc_xy[j, 0]] * 2, [arc_xy[j, 1]] * 2, [z_win[j], own_win[j]],
              color="0.5", lw=0.5, alpha=0.55, zorder=3)

    ax.plot(arc_xy[:, 0], arc_xy[:, 1], z_win, color="#d1495b", lw=1.8,
            zorder=5, label="on the surface")
    ax.plot(arc_xy[:, 0], arc_xy[:, 1], own_win, color="#1baf7a", lw=1.8,
            zorder=6, label="the window's own emotion")
    ax.set_zlim(0.0, 1.0)
    ax.set_xticklabels([]); ax.set_yticklabels([])
    ax.set_title(f"{e}   mean gap {np.abs(z_win - own_win).mean():.3f}", fontsize=10)
    ax.set_zlabel(e, fontsize=8)
    if i == 0:
      ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
  fig.suptitle(
    f"Both heights over the same {len(arc_xy)} windows -- {args.book} / {args.model}\n"
    f"red rides the terrain (height from location); green is what the text "
    f"carried; the stems are where the two disagree", fontsize=13)
  fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.02,
                      wspace=0.05, hspace=0.12)
  p = os.path.join(out_dir, paths.named("arc_on_surface", "paired"))
  fig.savefig(p, dpi=170); plt.close(fig); print(f"  wrote {p}")

  # --- figure 4: what the terrain cost the route ------------------------------
  fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.4))
  names = [m["emotion"] for m in metrics]
  a1.bar(names, [m["climb_ratio"] for m in metrics], color="#4a3aa7", alpha=0.85)
  a1.axhline(1.0, color="#33322e", lw=1)
  a1.set_ylabel("surface length / projected length")
  a1.set_title("What climbing cost the route\n(1.0 = the terrain is flat)", fontsize=10)
  a1.tick_params(axis="x", rotation=30)

  det = [m["excess_over_geodesic"] for m in metrics]
  if any(d is not None for d in det):
    a2.bar(names, [d if d else 0 for d in det], color="#1baf7a", alpha=0.85)
    a2.axhline(1.0, color="#33322e", lw=1)
    a2.set_ylim(bottom=0.98)
    a2.set_ylabel("route / geodesic, summed over legs")
    a2.set_title("Did it climb more than it had to?\n(1.0 = every step took the "
                 "geodesic)", fontsize=10)
    a2.tick_params(axis="x", rotation=30)
  sup_pct = 100 * float(np.mean([m["supported_fraction"] for m in metrics]))
  fig.suptitle(f"The story's route against the geometry (alpha {args.alpha})\n"
               f"measured only where the surface is supported "
               f"({sup_pct:.0f}% of the route on average)", fontsize=12)
  fig.tight_layout(rect=(0, 0, 1, 0.92))
  p = os.path.join(out_dir, paths.named("arc_on_surface", "cost"))
  fig.savefig(p, dpi=180); plt.close(fig); print(f"  wrote {p}")

  print(f"  -> {out_dir}")


if __name__ == "__main__":
  main()
