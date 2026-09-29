"""
Geodesics on the emotion landscape.

A geodesic is the shortest path between two points *along the surface*. Because
the surface is embedded in R^3 with the score as height, climbing costs distance,
so the shortest path bends around hills instead of over them. That bend is the
whole point of the visualization: a straight line in the latent plane ignores the
emotion, a geodesic is shaped by it.

The paths are computed *exactly*, by Kirsanov's implementation of the
Mitchell-Mount-Papadimitriou algorithm (the vendored C++ in geodesic_cpp/,
driven through _geodesic_native). Unlike a path restricted to the mesh's own
edges, an exact geodesic may cross a triangle's interior at any angle, so on a
flat surface it comes out genuinely straight -- the cheapest correctness check
in the whole pipeline.
"""

import numpy as np
from scipy.sparse.csgraph import connected_components

from geometry._geodesic_native import exact_geodesic_paths


def exact_paths(vertices, faces, pairs, graph=None):
  """Exact geodesic paths (Kirsanov / MMP), one per (source, target) pair.

  Returns a list of (m, 3) polylines running from source to target, or None
  where no path exists. The solver is the exact C++ implementation in
  geodesic_cpp/: a path may cross a triangle's interior at any angle, rather
  than being restricted to the mesh's edges.

  On a flat surface an exact geodesic is an actually straight line, which is the
  control the edge-restricted Dijkstra path fails.
  """
  V = np.asarray(vertices, dtype=np.float64)
  pairs = [(int(s), int(d)) for s, d in pairs]

  # The masked field can leave the mesh in disconnected pieces, and asking for a
  # path between two of them is not a question with an answer. Screening pairs
  # here keeps the solver off cross-component work and matches its own report of
  # an unreachable target (an empty path -> None).
  if graph is not None:
    _, comp = connected_components(graph, directed=False)
  else:
    comp = None

  # Solve only the answerable pairs in one batched call; splice None back in for
  # the cross-component ones so the returned list still lines up with `pairs`.
  solvable = [i for i, (s, d) in enumerate(pairs)
              if comp is None or comp[s] == comp[d]]
  solved = exact_geodesic_paths(V, faces, [pairs[i] for i in solvable])

  paths = [None] * len(pairs)
  for i, p in zip(solvable, solved):
    paths[i] = p
  return paths


def polyline_length(points):
  """Length of a polyline in R^3."""
  if points is None or len(points) < 2:
    return 0.0
  return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def polyline_deviation(points):
  """How far a polyline wanders from the straight line, in the (x, y) plane.

  The largest perpendicular distance from the path to the segment joining its
  endpoints, as a fraction of that segment's length. 0 means the path ignored
  the terrain; larger means the landscape pushed it aside.

  Measured in the plane on purpose: the question is whether the path takes a
  different *route*, not whether it goes up and down.
  """
  if points is None or len(points) < 3:
    return 0.0
  p = np.asarray(points)[:, :2]
  a, b = p[0], p[-1]
  ab = b - a
  span = np.linalg.norm(ab)
  if span < 1e-12:
    return 0.0
  perp = np.abs(np.cross(np.broadcast_to(ab, p.shape), p - a)) / span
  return float(perp.max() / span)


def rising_pairs(scores, min_delta=0.0):
  """Consecutive paragraphs i -> i+1 whose score increases.

  These are the arrows that "point toward" the emotion. min_delta guards against
  reading meaning into ties: the annotations are quantized to a handful of
  values, so a rise of exactly 0 is common and a rise of 0.05 may be the
  annotator's rounding rather than the story.
  """
  delta = np.diff(scores)
  idx = np.where(delta > min_delta)[0]
  return idx, delta[idx]
