"""Screened Poisson reconstruction of a closed emotion landscape.

Every other fit in this repo (`surface/bspline`, the kernel smoothers) recovers a
*height field*: one z per (x, y), an open sheet hanging over the PCA plane.
Poisson reconstruction cannot do that and does not try. It
takes oriented points -- a position and an outward normal -- and returns the
boundary of the solid those normals bound: watertight, closed, two-sided. So the
object here is not `z = f(x, y)`. It is the *solid under the landscape*,

    {(x, y, z) : (x, y) in the book's support, 0 <= z <= f(x, y)}

and what gets written out is its skin. That is why the folder is
`surface/poisson/` and not another entry beside `bspline_ls`.

The consequence is that the paragraph points alone are not a valid input. They
sample the top of that solid and nothing else, and a Poisson solve given only a
lid invents whatever sides and bottom the boundary condition implies -- usually
a bulge that runs off below the data and closes somewhere meaningless. So this
module builds the rest of the boundary explicitly:

    lid     the paragraphs, normals from a local plane fit, pointing up
    floor   z = 0 over the support region, normals pointing down
    walls   the rim of the support region, normals pointing outward

Only the lid carries text. The floor and walls are scaffolding that makes the
question well posed, and `fit_surface.py` reports the three counts separately so
a reader knows how much of the mesh is Alice and how much is closure.

Nothing here calls the vendored code as a library: PoissonRecon is a binary, and
`reconstruct()` shells out to it through PLY files on disk.
"""

import os
import subprocess
import sys

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))

BINARY = os.path.join(HERE, "vendor", "PoissonRecon", "Bin", "Linux",
                      "PoissonRecon")


# ---------------------------------------------------------------- local fit

def local_plane(X, z, Q, h):
  """Weighted local linear fit of z over (x, y), evaluated at every point of Q.

  Returns `(value, gradient)`: the fitted height at each query point and the
  (dz/dx, dz/dy) of the plane fitted there. Both come out of the same solve,
  which is the reason this is here rather than a call into
  `geometry.smoothers.local_linear` -- that returns the value and throws the
  slope away, and the slope is exactly what a normal is made of.

  `h` is an isotropic bandwidth in the units of X. Gaussian weights, no
  truncation: with ~800 paragraphs the dense matrix is small enough that a
  kd-tree would be ceremony.
  """
  X, z, Q = np.asarray(X, float), np.asarray(z, float), np.asarray(Q, float)
  value = np.empty(len(Q))
  grad = np.empty((len(Q), 2))

  for i, q in enumerate(Q):
    d = X - q
    w = np.exp(-0.5 * (d[:, 0] ** 2 + d[:, 1] ** 2) / h ** 2)
    # A query point far outside the cloud gets no weight at all and the normal
    # equations go singular. Falling back to the flat weighted mean there is
    # not a fit, but nothing queries there except the wall sampler at the very
    # rim of the mask, where the alternative is a NaN in the point cloud.
    if w.sum() < 1e-12:
      value[i], grad[i] = z.mean(), 0.0
      continue
    A = np.column_stack([np.ones(len(X)), d[:, 0], d[:, 1]])
    Aw = A * w[:, None]
    try:
      beta = np.linalg.solve(A.T @ Aw, Aw.T @ z)
    except np.linalg.LinAlgError:
      value[i], grad[i] = float(np.average(z, weights=w)), 0.0
      continue
    value[i], grad[i] = beta[0], beta[1:]

  return value, grad


def upward_normals(grad, relief):
  """Unit normals of the graph z = f(x, y), from its gradient.

  `relief` is the factor the height axis was scaled by when the cloud was put
  into the unit box; the normal has to be taken in the scaled space, or a
  landscape drawn twice as tall would carry the normals of the short one.
  """
  n = np.column_stack([-relief * grad[:, 0], -relief * grad[:, 1],
                       np.ones(len(grad))])
  return n / np.linalg.norm(n, axis=1, keepdims=True)


# ------------------------------------------------------------ support region

def support_grid(X, radius, resolution, pad=0.03):
  """A boolean grid: True where some paragraph is within `radius`.

  Same rule `surface/bspline` masks its plots with -- the region the book actually
  occupies, as opposed to the bounding rectangle it happens to span. Here it is
  load-bearing rather than cosmetic: its interior is where the floor goes and
  its rim is where the walls go, so it *is* the footprint of the solid.
  """
  X = np.asarray(X, float)
  lo, hi = X.min(axis=0), X.max(axis=0)
  span = hi - lo
  lo, hi = lo - pad * span, hi + pad * span

  gx, gy = np.meshgrid(np.linspace(lo[0], hi[0], resolution),
                       np.linspace(lo[1], hi[1], resolution))
  cells = np.column_stack([gx.ravel(), gy.ravel()])

  keep = np.zeros(len(cells), dtype=bool)
  for start in range(0, len(cells), 4096):
    block = cells[start:start + 4096]
    d = np.linalg.norm(block[:, None, :] - X[None, :, :], axis=2)
    keep[start:start + 4096] = d.min(axis=1) <= radius

  mask = keep.reshape(gx.shape)
  # One erosion-dilation round drops isolated cells and pinholes. A one-cell
  # island would otherwise become its own sealed component in the mesh, and a
  # pinhole a tunnel through the solid -- both watertight, neither meaningful.
  mask = ndimage.binary_opening(mask, np.ones((3, 3)))
  mask = ndimage.binary_fill_holes(mask)
  return gx, gy, mask


def rim_points(gx, gy, mask, sigma=2.0):
  """The outer boundary cells of `mask`, and the outward horizontal direction.

  Outward is minus the gradient of the blurred mask: the blur turns the 0/1
  indicator into a ramp that falls as it crosses the boundary, so its downhill
  direction is out. The blur is what makes this work on a ragged rim -- the
  support region's boundary steps cell by cell, and any direction read off a
  single step is one of eight, at 45 degrees to what the rim is actually doing.
  `sigma` is in cells: 2 averages the rim over a five-cell neighbourhood.

  The distance transform is the other obvious source for this and is worse here:
  it is flat one cell in from the boundary, which is exactly where the rim is.
  """
  edge = mask & ~ndimage.binary_erosion(mask, np.ones((3, 3)))
  ramp = ndimage.gaussian_filter(mask.astype(float), sigma, mode="constant")
  # np.gradient's first output varies along axis 0. Under meshgrid's default
  # "xy" indexing that axis is y, and axis 1 is x -- the transposition that
  # makes this line worth spelling out.
  d_dy, d_dx = np.gradient(ramp)
  n = np.column_stack([-d_dx[edge], -d_dy[edge]])
  norm = np.linalg.norm(n, axis=1, keepdims=True)
  # A cell where the ramp is flat -- deep inside a blur-filled notch -- has no
  # direction to offer, and is dropped rather than given an arbitrary one.
  ok = norm[:, 0] > 1e-9
  return np.column_stack([gx[edge], gy[edge]])[ok], n[ok] / norm[ok]


def merge_coincident(X, z, radius):
  """Collapse paragraphs that land on the same spot, averaging their scores.

  PCA is a projection, and it is not injective: 144 of Alice's paragraph pairs
  land within 0.005 of each other in the unit box and some land on exactly the
  same point, carrying different scores. Two heights over one (x, y) is not a
  height field, and a reconstruction handed both does the only thing it can --
  it builds a surface that passes through both, which means a chimney plunging
  from the upper score to the lower one and back. A row of those along a ridge
  is the comb that shows up in the rendered landscape, and each one is a tunnel
  through the solid, which is where the Euler characteristic loses its 2.

  Averaging is a choice and it loses something real: those paragraphs genuinely
  differ, and the projection is what threw the difference away. What it buys is
  a lid that is a function, which is the precondition for the question "what is
  the height here" to have an answer at all.

  Returns `(X, z, merged)` where `merged` is how many points disappeared.
  """
  X, z = np.asarray(X, float), np.asarray(z, float)
  if radius <= 0:
    return X, z, 0

  cell = np.round(X / radius).astype(np.int64)
  _, inverse, counts = np.unique(cell, axis=0, return_inverse=True,
                                 return_counts=True)
  if len(counts) == len(X):
    return X, z, 0

  # The merged point sits at the centroid of the group, not at the cell centre:
  # the grid is only how coincidence is detected, not where the answer lives.
  n = len(counts)
  merged_x = np.zeros((n, 2))
  np.add.at(merged_x, inverse, X)
  merged_z = np.bincount(inverse, weights=z, minlength=n)
  return (merged_x / counts[:, None], merged_z / counts, len(X) - n)


# ------------------------------------------------- extending past the support

def extend_heights(X, z, Q, bandwidth, k=25):
  """The fitted height at query points that may lie outside the point cloud.

  Nadaraya-Watson with a bandwidth that grows to reach the k nearest paragraphs,
  which is what makes it usable off the end of the data: at a query inside the
  cloud it is an ordinary kernel smooth at `bandwidth`, and at one far outside
  the weights are dominated by the nearest handful, so the estimate flattens to
  their average instead of running off.

  That flattening is the whole reason this is not `local_plane`. A local *linear*
  fit extrapolates its slope, and a slope fitted at the edge of the cloud, run
  out to the corner of the rectangle, leaves the [0, 1] a score lives in -- the
  surface would rise to a score of 3 over a region containing no paragraphs. A
  constant extension is also wrong out there, but it is wrong within the range
  the data occupies, and it is obviously an extension rather than a claim.

  Nothing here makes the region outside the support meaningful. It is filled
  because the caller asked for a surface over the whole plane; `fit_surface.py`
  keeps the support mask beside it so a reader can see which is which.
  """
  X, z, Q = np.asarray(X, float), np.asarray(z, float), np.asarray(Q, float)
  k = min(k, len(X))
  d, idx = cKDTree(X).query(Q, k=k)
  d, idx = np.atleast_2d(d), np.atleast_2d(idx)
  # Half the distance to the kth neighbour, so the k points that set the
  # bandwidth are also the ones that carry most of the weight.
  h = np.maximum(bandwidth, d[:, -1:] / 2)
  w = np.exp(-0.5 * (d / h) ** 2)
  return (w * z[idx]).sum(axis=1) / w.sum(axis=1)


def lid_fill(X, z, gx, gy, support, relief, bandwidth, stride, buffer_cells=0):
  """Lid samples on the grid *outside* the support, with normals.

  Inside the support the lid is the paragraphs themselves; this is only what
  covers the rest of the rectangle. Normals come from finite differences of the
  filled height grid rather than from a per-point fit, because a normal read off
  the same grid the points sit on is guaranteed consistent with its neighbours
  -- and out here, where there is no data to be faithful to, consistency is the
  only property left worth having.

  `buffer_cells` holds the fill back that many cells from the support, and it is
  not cosmetic. The two lids disagree about height wherever they meet: inside,
  the lid passes through raw paragraph scores, which are noisy; outside, through
  a kernel estimate, which is smooth. Sampled right up against each other, the
  solver is handed two contradictory heights a cell apart, and it resolves the
  contradiction by tearing -- a row of slots down through the terrain along the
  support boundary, which look like the teeth of a comb. Left a gap, it
  interpolates between them instead, which is what it is good at.
  """
  Q = np.column_stack([gx.ravel(), gy.ravel()])
  H = np.clip(extend_heights(X, z, Q, bandwidth), 0.0, 1.0).reshape(gx.shape)

  dx = gx[0, 1] - gx[0, 0]
  dy = gy[1, 0] - gy[0, 0]
  d_dy, d_dx = np.gradient(H, dy, dx)

  near = support
  if buffer_cells > 0:
    near = ndimage.binary_dilation(support, np.ones((3, 3)),
                                   iterations=int(buffer_cells))

  keep = np.zeros_like(support)
  keep[::stride, ::stride] = True
  sel = keep & ~near

  points = np.column_stack([gx[sel], gy[sel], relief * H[sel]])
  normals = np.column_stack([-relief * d_dx[sel], -relief * d_dy[sel],
                             np.ones(int(sel.sum()))])
  normals /= np.linalg.norm(normals, axis=1, keepdims=True)
  return points, normals, H


# -------------------------------------------------------------- point cloud

class Cloud:
  """Oriented points, tagged by which part of the boundary they came from."""

  def __init__(self, points, normals, part):
    self.points = np.asarray(points, float)
    self.normals = np.asarray(normals, float)
    self.part = np.asarray(part)
    # Set by build_cloud: the support mask, and the extended height grid when
    # there is one. Both are about telling fit from extension downstream.
    self.support = None
    self.height = None
    self.merged = 0

  def counts(self):
    return {p: int((self.part == p).sum()) for p in ("lid", "floor", "wall")}


def build_cloud(X, z, relief, radius, resolution, bandwidth, wall_spacing,
                floor_stride, base=0.0, footprint="support", fill_buffer=0.5,
                merge_radius=0.0, lid="fitted"):
  """The closed oriented boundary of the solid under the emotion landscape.

  `X` is the PCA plane already rescaled so the longer axis spans [0, 1] --
  `fit_surface.normalize` does that -- and `z` is the emotion score in [0, 1].
  `relief` multiplies z, so `relief=1` gives a landscape as tall as the book is
  wide and `relief=0.3` a flatter one. It is a real choice, not a display
  setting: Poisson's octree is isotropic, so the aspect ratio decides how much
  resolution the vertical structure gets.

  `lid` decides what height the paragraph samples sit at:

    fitted   the local plane fit's own value, so the lid is a smooth surface
             and `bandwidth` is the smoothing knob, as it is in every sibling
             fit in this repo.
    scores   the raw per-paragraph score. Screened Poisson interpolates its
             samples, and adjacent paragraphs disagree by as much as 0.8 -- so
             the reconstruction plunges a narrow canyon between every such pair
             and the landscape comes out combed. That is not a defect in the
             solver, it is what interpolating this data looks like, and it is
             kept as an option because seeing it is the argument for smoothing.

  `base` drops the floor to `z = -base`, putting a plinth of that thickness
  under the whole landscape. Without one, a solid whose lid is a score of 0.0
  has zero thickness there, the lid and floor land in the same octree cell, and
  the reconstruction pinches through -- which is how a "closed surface" acquires
  a tunnel and stops being the boundary of anything. Over half of Alice's
  paragraphs score sadness below 0.1, so this is the common case, not the edge
  case. The plinth is scaffolding like the floor: it carries no text, and z = 0
  is still where a score of zero sits.

  `footprint` decides what the solid stands on:

    support   the region the book occupies -- a ragged outline, and the honest
              one, since the lid is only ever over paragraphs.
    box       the whole PCA rectangle. The lid is extended past the support by
              `lid_fill` so it covers the plane, which is what `--open` wants:
              a surface over all of (PC1, PC2) rather than over the outline.
              Everything outside the support is an extension of the fit, not a
              fit; `Cloud.support` records where the difference is.
  """
  X = np.asarray(X, float)
  gx, gy, support = support_grid(X, radius, resolution)
  if footprint == "box":
    mask = np.ones_like(support)
  elif footprint == "support":
    mask = support
  else:
    raise ValueError(f"footprint must be 'support' or 'box', not {footprint!r}")

  # -- lid: the paragraphs themselves, the only part the text touches. Merged
  # first where the projection put two of them on one spot, since the lid has
  # to be a function of (x, y) before it can be a surface over it.
  Xl, zl, merged = merge_coincident(X, z, merge_radius)
  fitted, grad = local_plane(Xl, zl, Xl, bandwidth)
  if lid == "fitted":
    zl_used = np.clip(fitted, 0.0, 1.0)
  elif lid == "scores":
    zl_used = zl
  else:
    raise ValueError(f"lid must be 'fitted' or 'scores', not {lid!r}")
  lid_pts = np.column_stack([Xl, relief * zl_used])
  lid_n = upward_normals(grad, relief)

  # ...plus, over a box footprint, the rest of the rectangle.
  height = None
  if footprint == "box":
    # The gap is measured in support radii, so it scales with how far apart the
    # paragraphs are rather than with the grid resolution.
    buffer_cells = round(fill_buffer * radius / abs(gx[0, 1] - gx[0, 0]))
    fill_pts, fill_n, height = lid_fill(Xl, zl, gx, gy, support, relief,
                                        bandwidth, floor_stride, buffer_cells)
    lid_pts = np.vstack([lid_pts, fill_pts])
    lid_n = np.vstack([lid_n, fill_n])

  # -- floor: z = 0 under the footprint, thinned so it does not out-vote the lid.
  ii, jj = np.nonzero(mask)
  sel = (ii % floor_stride == 0) & (jj % floor_stride == 0)
  floor_xy = np.column_stack([gx[ii[sel], jj[sel]], gy[ii[sel], jj[sel]]])
  floor_pts = np.column_stack([floor_xy, np.full(len(floor_xy), -base)])
  floor_n = np.tile([0.0, 0.0, -1.0], (len(floor_pts), 1))

  # -- walls: from the floor up to the lid, at every rim cell.
  rim_xy, rim_dir = rim_points(gx, gy, mask)
  if footprint == "box":
    # The rim of the rectangle is the one place the walls are certainly outside
    # the support, where a local linear fit has no data and extrapolates a
    # slope. This is the same extension the lid got, so wall and lid agree on
    # where they meet.
    rim_h = extend_heights(Xl, zl, rim_xy, bandwidth)
  else:
    rim_h, _ = local_plane(Xl, zl, rim_xy, bandwidth)
  rim_h = relief * np.clip(rim_h, 0.0, 1.0)

  wall_pts, wall_n = [], []
  for (x, y), (nx, ny), h in zip(rim_xy, rim_dir, rim_h):
    # At least the two endpoints, so a wall of zero height still seals the
    # seam between floor and lid rather than leaving a crack for the solver.
    steps = max(2, int(np.ceil((h + base) / wall_spacing)) + 1)
    for t in np.linspace(-base, h, steps):
      wall_pts.append((x, y, t))
      wall_n.append((nx, ny, 0.0))

  points = np.vstack([lid_pts, floor_pts, np.array(wall_pts)])
  normals = np.vstack([lid_n, floor_n, np.array(wall_n)])
  part = np.array(["lid"] * len(lid_pts) + ["floor"] * len(floor_pts) +
                  ["wall"] * len(wall_pts))
  cloud = Cloud(points, normals, part)
  cloud.support = support
  cloud.height = height
  cloud.merged = merged
  return cloud, (gx, gy, mask)


# ------------------------------------------------------------------ PLY I/O

_PLY_NP = {
  "float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8",
  "int": "<i4", "int32": "<i4", "uint": "<u4", "uint32": "<u4",
  "short": "<i2", "int16": "<i2", "ushort": "<u2", "uint16": "<u2",
  "char": "<i1", "int8": "<i1", "uchar": "<u1", "uint8": "<u1",
}


def write_ply(path, points, normals):
  """Ascii PLY with normals. The input side is a few thousand points; ascii is
  cheap here and leaves the file readable when a reconstruction looks wrong."""
  with open(path, "w") as f:
    f.write("ply\nformat ascii 1.0\n")
    f.write(f"element vertex {len(points)}\n")
    for c in "xyz":
      f.write(f"property float {c}\n")
    for c in "xyz":
      f.write(f"property float n{c}\n")
    f.write("end_header\n")
    for p, n in zip(points, normals):
      f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} "
              f"{n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")


def read_ply(path):
  """Vertices and triangles from a PLY, ascii or binary little-endian.

  Enough of the format for what PoissonRecon writes -- vertices with float or
  double coordinates plus whatever extra per-vertex properties were asked for,
  and a face list of variable-length integer lists. Faces with more than three
  corners are fanned into triangles.
  """
  with open(path, "rb") as f:
    if f.readline().strip() != b"ply":
      raise ValueError(f"{path} is not a PLY file")

    fmt, elements = None, []
    while True:
      line = f.readline().decode("ascii", "replace").strip()
      if not line:
        raise ValueError(f"{path} ended inside its header")
      tok = line.split()
      if tok[0] == "format":
        fmt = tok[1]
      elif tok[0] == "element":
        elements.append((tok[1], int(tok[2]), []))
      elif tok[0] == "property":
        elements[-1][2].append(tok[1:])
      elif tok[0] == "end_header":
        break

    if fmt == "ascii":
      return _read_ply_ascii(f, elements)
    if fmt != "binary_little_endian":
      raise ValueError(f"Unsupported PLY format {fmt!r} in {path}")
    return _read_ply_binary(f, elements)


def _faces_from_lists(lists):
  tris = []
  for idx in lists:
    for k in range(1, len(idx) - 1):
      tris.append((idx[0], idx[k], idx[k + 1]))
  return np.array(tris, dtype=np.int64).reshape(-1, 3)


def _read_ply_ascii(f, elements):
  verts, faces = None, None
  for name, count, props in elements:
    rows = [f.readline().split() for _ in range(count)]
    if name == "vertex":
      names = [p[-1] for p in props]
      cols = [names.index(c) for c in "xyz"]
      verts = np.array([[float(r[c]) for c in cols] for r in rows])
    elif name == "face":
      faces = _faces_from_lists([[int(v) for v in r[1:1 + int(r[0])]]
                                 for r in rows])
  return verts, faces


def _read_ply_binary(f, elements):
  verts, faces = None, None
  for name, count, props in elements:
    if props[0][0] == "list":
      # Variable-length rows: read one at a time, there is no fixed stride.
      count_t = np.dtype(_PLY_NP[props[0][1]])
      item_t = np.dtype(_PLY_NP[props[0][2]])
      lists = []
      for _ in range(count):
        n = int(np.frombuffer(f.read(count_t.itemsize), count_t)[0])
        lists.append(np.frombuffer(f.read(n * item_t.itemsize), item_t))
      if name == "face":
        faces = _faces_from_lists(lists)
      continue

    dtype = np.dtype([(p[-1], _PLY_NP[p[0]]) for p in props])
    rows = np.frombuffer(f.read(count * dtype.itemsize), dtype, count)
    if name == "vertex":
      verts = np.column_stack([rows["x"], rows["y"], rows["z"]]).astype(float)
  return verts, faces


# ------------------------------------------------------------ the vendor call

def reconstruct(points, normals, work_dir, depth=8, point_weight=4.0,
                samples_per_node=1.5, binary=BINARY, extra=(), verbose=False,
                retries=4):
  """Run the vendored PoissonRecon on an oriented cloud. Returns (verts, faces).

  `work_dir` keeps the two PLY files the binary talks through. They are kept
  rather than deleted: the input is the thing to look at first when a mesh comes
  out wrong, and it is small.

  **The retries.** The vendor's iso-surface extractor sometimes aborts with
  "Failed to close loop" -- a degeneracy where the level set passes exactly
  through an octree corner and the marching cubes case is ambiguous. It is not a
  bad cloud: the same points reconstruct fine at a slightly different depth, or
  scale, or with the whole cloud nudged. This cloud invites it, because floors
  and walls sampled from a regular grid put a great many samples on exactly
  coincident planes.

  So a failure is retried with the cloud jittered by a fraction of a cell. That
  is a real change to the input, and a small one -- 2% of a cell, against a lid
  whose own residual runs to whole cells -- but it is not nothing, so the jitter
  is seeded from the attempt number, reported, and recorded in the returned
  attempt count rather than hidden.
  """
  if not os.path.exists(binary):
    raise FileNotFoundError(
      f"PoissonRecon binary not found at {binary}. "
      f"Build it with surface/poisson/build_vendor.sh"
    )

  in_ply = os.path.join(work_dir, "cloud.ply")
  out_ply = os.path.join(work_dir, "mesh.ply")
  cell = 1.0 / 2 ** depth

  for attempt in range(retries + 1):
    if attempt == 0:
      sent = points
    else:
      rng = np.random.default_rng(attempt)
      sent = points + rng.normal(scale=0.02 * cell, size=np.shape(points))
    write_ply(in_ply, sent, normals)

    cmd = [binary, "--in", in_ply, "--out", out_ply,
           "--depth", str(depth),
           "--pointWeight", str(point_weight),
           "--samplesPerNode", str(samples_per_node),
           # Neumann is the vendor default and the right one here: the solid
           # runs up to the edge of its own bounding box on the floor, and
           # Dirichlet would pull the indicator to zero there and round the
           # base off.
           "--bType", "3",
           *extra]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
      break
    if "Failed to close loop" not in result.stderr or attempt == retries:
      raise RuntimeError(
        f"PoissonRecon failed ({result.returncode}) on attempt {attempt + 1}\n"
        f"  {' '.join(cmd)}\n{result.stdout}\n{result.stderr}"
      )
    print(f"  iso-surface extraction hit a degenerate cell; retrying with the "
          f"cloud jittered by {0.02 * cell:.2e} (attempt {attempt + 2})")

  if verbose:
    print(result.stdout.rstrip())

  verts, faces = read_ply(out_ply)
  if verts is None or faces is None:
    raise RuntimeError(f"PoissonRecon wrote no mesh to {out_ply}")
  return verts, faces


# ------------------------------------------------------------------ measures

def components(verts, faces):
  """Face indices grouped by connected component, largest first.

  A book is one solid, so a reconstruction with more than one component has
  produced something spurious -- usually a bubble a few cells across, thrown off
  where the wall sampling is dense enough that a sliver of the octree solves to
  "inside" on its own. `keep_largest` removes them; this is what says how big
  they were before they went.
  """
  parent = np.arange(len(verts))

  def find(i):
    while parent[i] != i:
      parent[i] = parent[parent[i]]
      i = parent[i]
    return i

  for a, b, c in faces:
    ra, rb, rc = find(a), find(b), find(c)
    parent[rb] = ra
    parent[rc] = ra

  root = np.array([find(i) for i in faces[:, 0]])
  groups = [np.nonzero(root == r)[0] for r in np.unique(root)]
  return sorted(groups, key=len, reverse=True)


def keep_largest(verts, faces):
  """The largest connected component, reindexed. Returns (verts, faces, dropped).

  `dropped` is a list of the triangle counts thrown away, so a caller can print
  "one 74-triangle bubble removed" rather than silently returning a different
  mesh than the solver produced.
  """
  groups = components(verts, faces)
  if len(groups) == 1:
    return verts, faces, []

  keep = faces[groups[0]]
  used = np.unique(keep)
  remap = np.full(len(verts), -1, dtype=np.int64)
  remap[used] = np.arange(len(used))
  return verts[used], remap[keep], [len(g) for g in groups[1:]]


def face_normals(verts, faces):
  """Unit normals, outward for a consistently oriented closed mesh."""
  a, b, c = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
  n = np.cross(b - a, c - a)
  return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-30)


def top_sheet(verts, faces, min_nz=0.1):
  """The lid alone: the open surface on top of the closed solid.

  A closed reconstruction is not a height field, and cannot be turned into one
  by wishing. What it does have is one face of the boundary that *is* the
  landscape -- the part whose outward normal points up -- and everything else is
  the scaffolding that closed it: a floor pointing down, and walls pointing
  sideways. `min_nz` is where the cut goes, on the normal's z component: 0.1
  keeps anything less than 84 degrees from horizontal, so an overhanging fold of
  the terrain survives and a wall does not.

  The largest connected component of what survives is the sheet; smaller ones
  are shelves on the outside of the walls, and are dropped. Returns
  `(verts, faces, dropped)` like `keep_largest`, and the result has a boundary
  -- it is open, and `is_watertight` on it is correctly False.
  """
  up = face_normals(verts, faces)[:, 2] > min_nz
  if not up.any():
    raise ValueError("No upward-facing triangles; is the mesh oriented inward?")

  kept = faces[up]
  groups = components(verts, kept)
  sheet = kept[groups[0]]

  used = np.unique(sheet)
  remap = np.full(len(verts), -1, dtype=np.int64)
  remap[used] = np.arange(len(used))
  return verts[used], remap[sheet], [len(g) for g in groups[1:]]


def upper_envelope(verts, faces, gx, gy):
  """The height of the top of the solid over every cell of the grid.

  What `top_sheet` returns is the honest object -- the actual triangles on top
  of the reconstruction -- and it is not a height field: it has folds, and once
  its overhanging faces are cut away it has holes where they were. This is the
  other thing you can ask for, and the one a surface over the whole plane needs:
  the *upper envelope*, one height per column, hole-free by construction.

  Every triangle is rasterized onto the grid and the maximum kept, so this is
  the exact top of the solid: the highest point of the surface over a cell lies
  on some triangle, and every triangle is drawn. No normal test is needed --
  floor and walls lose to the lid in their own column by being underneath it.

  Rasterizing rather than binning the vertices is the whole point, and the
  difference is not subtle. A cliff column can easily hold no lid *vertex* near
  its top -- the plateau above is one large triangle whose corners are
  elsewhere -- while holding plenty of lid *surface*. Binning vertices puts such
  a column at the foot of the cliff, cutting a one-cell slot straight down
  through the terrain, and a run of them along a cliff edge looks like a comb.

  Where the sheet does not fold this is the sheet, resampled. Where it does, it
  is the top of the fold, and `folded_fraction` says how much of the plane that
  concerns.
  """
  x0, y0 = gx[0, 0], gy[0, 0]
  dx, dy = gx[0, 1] - gx[0, 0], gy[1, 0] - gy[0, 0]
  ny, nx = gx.shape
  Z = np.full(gx.shape, -np.inf)

  tri = verts[faces]
  # Cell-index bounding box per triangle, clipped to the grid.
  lo = np.floor((tri[:, :, :2].min(axis=1) - [x0, y0]) / [dx, dy]).astype(int)
  hi = np.ceil((tri[:, :, :2].max(axis=1) - [x0, y0]) / [dx, dy]).astype(int)
  lo = np.maximum(lo, 0)
  hi = np.minimum(hi + 1, [nx, ny])

  for t, (j0, i0), (j1, i1) in zip(tri, lo, hi):
    if j0 >= j1 or i0 >= i1:
      continue
    (ax, ay, az), (bx, by, bz), (cx, cy, cz) = t
    det = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
    if abs(det) < 1e-14:      # edge-on: no interior to fill
      continue

    px, py = np.meshgrid(x0 + dx * np.arange(j0, j1),
                         y0 + dy * np.arange(i0, i1))
    u = ((by - cy) * (px - cx) + (cx - bx) * (py - cy)) / det
    v = ((cy - ay) * (px - cx) + (ax - cx) * (py - cy)) / det
    w = 1.0 - u - v
    # A hair of slack, so a cell centre landing exactly on a shared edge is
    # claimed by both triangles rather than by neither.
    inside = (u >= -1e-9) & (v >= -1e-9) & (w >= -1e-9)
    if not inside.any():
      continue

    block = Z[i0:i1, j0:j1]
    np.maximum(block, np.where(inside, u * az + v * bz + w * cz, -np.inf),
               out=block)

  # Only cells the mesh does not cover at all, which over a box footprint means
  # the odd cell whose centre falls in a crack between triangles.
  missed = ~np.isfinite(Z)
  if missed.any():
    from scipy.interpolate import griddata
    Z[missed] = griddata(np.column_stack([gx[~missed], gy[~missed]]),
                         Z[~missed],
                         np.column_stack([gx[missed], gy[missed]]),
                         method="nearest")
  return Z, float(missed.mean())


def grid_mesh(gx, gy, Z):
  """A grid of heights as a triangle mesh: two triangles per cell.

  So the open surface can be written as a mesh as well as a field, and read by
  anything that takes the closed fit's meshes. It has a boundary -- the rim of
  the rectangle -- which is the entire point.
  """
  ny, nx = Z.shape
  verts = np.column_stack([gx.ravel(), gy.ravel(), Z.ravel()])
  j, i = np.meshgrid(np.arange(nx - 1), np.arange(ny - 1))
  a = (i * nx + j).ravel()
  b, c, d = a + 1, a + nx, a + nx + 1
  faces = np.vstack([np.column_stack([a, c, b]),
                     np.column_stack([b, c, d])])
  return verts, faces


def rasterize(verts, gx, gy):
  """A sheet's height on the plotting grid, so it can be saved as a field.

  Linear interpolation over the sheet's own vertices, with nearest-neighbour
  fill for grid points the triangulation does not cover (the corners, mostly,
  where the sheet rolls over the rim). This is a resampling of a surface that
  already exists, not another fit -- the fit was the Poisson solve.

  Only meaningful for a sheet that really is a height field. `fit_surface.py`
  checks that before calling: if the reconstruction folded, one z per (x, y) is
  a lie and the field is not written.
  """
  from scipy.interpolate import griddata

  Q = np.column_stack([gx.ravel(), gy.ravel()])
  Z = griddata(verts[:, :2], verts[:, 2], Q, method="linear")
  holes = np.isnan(Z)
  if holes.any():
    Z[holes] = griddata(verts[:, :2], verts[:, 2], Q[holes], method="nearest")
  return Z.reshape(gx.shape)


def folded_fraction(verts, gx, gy, gap=3.0):
  """The fraction of covered grid cells the sheet passes over more than once.

  Whether `z = f(x, y)` is even true of this sheet, measured rather than
  assumed. Sheet vertices are binned by grid cell and sorted by height; a cell
  where consecutive heights jump by more than `gap` cells' worth is a cell the
  surface crosses twice, i.e. an overhang. A smooth steep slope does not
  trigger it -- its vertices are dense in z, however fast it climbs.

  0 means the sheet is a genuine height field and `rasterize` loses nothing.
  """
  dx = abs(gx[0, 1] - gx[0, 0])
  dy = abs(gy[1, 0] - gy[0, 0])
  ix = np.clip(((verts[:, 0] - gx[0, 0]) / dx).astype(int), 0, gx.shape[1] - 1)
  iy = np.clip(((verts[:, 1] - gy[0, 0]) / dy).astype(int), 0, gy.shape[0] - 1)
  cell = iy * gx.shape[1] + ix

  order = np.lexsort((verts[:, 2], cell))
  c, z = cell[order], verts[order, 2]
  same = c[1:] == c[:-1]
  jump = np.diff(z) > gap * max(dx, dy)

  folded = np.unique(c[1:][same & jump])
  occupied = np.unique(c)
  return float(len(folded) / max(1, len(occupied)))


def euler_characteristic(verts, faces):
  """V - E + F. 2 for a sphere-like closed surface; anything else has handles,
  boundary, or extra components -- all of which say the closure did not work."""
  edges = set()
  for a, b, c in faces:
    for u, v in ((a, b), (b, c), (c, a)):
      edges.add((u, v) if u < v else (v, u))
  return len(verts) - len(edges) + len(faces)


def is_watertight(faces):
  """True when every edge is shared by exactly two triangles."""
  seen = {}
  for a, b, c in faces:
    for u, v in ((a, b), (b, c), (c, a)):
      key = (u, v) if u < v else (v, u)
      seen[key] = seen.get(key, 0) + 1
  return all(n == 2 for n in seen.values())


def lid_residual(verts, lid_points):
  """Distance from each paragraph to the nearest mesh vertex.

  A proxy for point-to-surface distance, not the thing itself: it reads high by
  up to half an edge length. At depth 8 over a unit box the edges are ~1/256, so
  a residual reported in the hundredths is the surface genuinely missing the
  paragraph, and one in the thousandths is discretization.
  """
  d, _ = cKDTree(verts).query(lid_points)
  return d
