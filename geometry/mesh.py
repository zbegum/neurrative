"""
Turn a masked height field into a triangulated surface in R^3.

The scalar field gives z = f(x, y) on a grid. Lifting each grid point to
(x, y, alpha*f) and triangulating gives the landscape the geodesics walk on.

`alpha` is vertical exaggeration and it is the parameter that decides whether
this visualization has anything to show. Distances on the surface are measured in
R^3, so climbing costs length: a path prefers to go around a hill rather than
over it. How much it prefers that depends entirely on how tall the hill is
relative to the map -- at alpha = 0 the surface is flat and every geodesic is a
straight line. alpha is not cosmetic, it is the exchange rate between "score
units" and "PCA units", and nothing in the data sets it for us.
"""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def height_mesh(gx, gy, Z, mask, alpha):
  """Lift a masked grid to vertices and triangles.

  Returns (vertices (n,3), faces (m,3), vid (H,W)) where vid indexes each grid
  cell's vertex, or -1 where the field was masked out.
  """
  valid = mask & np.isfinite(Z)
  vid = np.full(Z.shape, -1, dtype=np.int64)
  vid[valid] = np.arange(int(valid.sum()))

  vertices = np.column_stack([
    gx[valid].ravel(), gy[valid].ravel(), alpha * Z[valid].ravel(),
  ])

  # Two triangles per grid cell, but only where all four corners survived the
  # density mask. Cells with three valid corners are dropped rather than
  # half-filled: they sit on the ragged edge of the masked region, and a sliver
  # triangle there would be a spike of geometry nobody asked for.
  a, b = vid[:-1, :-1], vid[:-1, 1:]
  c, d = vid[1:, :-1], vid[1:, 1:]
  ok = (a >= 0) & (b >= 0) & (c >= 0) & (d >= 0)

  faces = np.concatenate([
    np.stack([a[ok], b[ok], d[ok]], axis=1),
    np.stack([a[ok], d[ok], c[ok]], axis=1),
  ])

  # Every valid cell got a vertex, but cells on the ragged mask boundary may
  # belong to no surviving face. Those orphans are not part of the surface, and
  # a mesh solver is entitled to reject them, so drop them and renumber.
  used = np.unique(faces)
  remap = np.full(len(vertices), -1, dtype=np.int64)
  remap[used] = np.arange(len(used))
  vertices = vertices[used]
  faces = remap[faces]
  vid = np.where(vid >= 0, remap[np.clip(vid, 0, None)], -1)

  return vertices, faces, vid


def largest_component(vertices, faces):
  """Keep only the biggest edge-connected piece of the mesh.

  The exact solver requires an edge-connected mesh and *asserts* on anything
  else -- an abort inside the C++, not an exception you can catch. The density
  mask routinely fragments the field into islands (at hx = 0.15, resolution 120
  the wonder terrain comes out in three pieces), so the mesh has to be reduced
  before it is handed over.

  Dropping the small islands is the honest reduction: a geodesic between two
  points on different components does not exist, so those pieces could never
  have carried a path anyway. Returns (vertices, faces, kept_fraction) so the
  caller can report how much of the surface it is actually walking on.
  """
  n_comp, label = connected_components(edge_graph(vertices, faces),
                                       directed=False)
  if n_comp == 1:
    return vertices, faces, 1.0

  keep = label == int(np.argmax(np.bincount(label)))
  faces = faces[keep[faces].all(axis=1)]

  used = np.unique(faces)
  remap = np.full(len(vertices), -1, dtype=np.int64)
  remap[used] = np.arange(len(used))
  return vertices[used], remap[faces], float(len(used) / len(vertices))


def edge_graph(vertices, faces):
  """Symmetric sparse graph of triangle edges, weighted by length in R^3."""
  e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
  e = np.sort(e, axis=1)
  e = np.unique(e, axis=0)

  lengths = np.linalg.norm(vertices[e[:, 0]] - vertices[e[:, 1]], axis=1)
  n = len(vertices)

  # Both directions, so Dijkstra can traverse either way.
  rows = np.concatenate([e[:, 0], e[:, 1]])
  cols = np.concatenate([e[:, 1], e[:, 0]])
  data = np.concatenate([lengths, lengths])
  return coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()


def snap(points, vertices):
  """Nearest surviving vertex to each point, by (x, y) only.

  Returns (indices, distances). A large distance means the point fell in a
  region the density mask removed, so its foot on the surface is a guess --
  callers should report those rather than silently walk from the wrong place.
  """
  from scipy.spatial import cKDTree
  tree = cKDTree(vertices[:, :2])
  dist, idx = tree.query(np.asarray(points)[:, :2])
  return idx, dist
