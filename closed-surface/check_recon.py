"""What holds the vendor call honest. `python closed-surface/check_recon.py`

There is no port to check here -- the solver is the vendor's own binary -- so
what is checkable is everything on either side of it: the PLY we write, the PLY
we read back, the normals we hand it, and whether a shape whose answer is known
comes back as that shape.

Four checks:

  ply roundtrip   points and normals survive write -> read unchanged
  sphere          a sampled sphere reconstructs to a sphere: watertight, Euler
                  characteristic 2, radius right to a fraction of a cell
  normals         `local_plane` recovers the gradient of a known tilted plane,
                  and `upward_normals` turns it into the plane's own normal
  box             the floor/wall/lid machinery on a flat synthetic landscape
                  gives a watertight solid of the right volume
"""

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import poisson

FAILURES = []


def check(name, ok, detail=""):
  print(f"  {'ok  ' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
  if not ok:
    FAILURES.append(name)


def mesh_volume(verts, faces):
  """Signed volume by the divergence theorem, one tetrahedron per triangle."""
  a, b, c = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
  return float(abs(np.einsum("ij,ij->i", a, np.cross(b, c)).sum()) / 6.0)


def check_ply_roundtrip():
  print("\nply roundtrip")
  rng = np.random.default_rng(0)
  pts = rng.random((50, 3))
  n = rng.normal(size=(50, 3))
  n /= np.linalg.norm(n, axis=1, keepdims=True)

  with tempfile.TemporaryDirectory() as d:
    path = os.path.join(d, "p.ply")
    poisson.write_ply(path, pts, n)
    back, faces = poisson.read_ply(path)

  # write_ply prints 6 decimals, so agreement is to 1e-6, not to float equality.
  check("positions survive", np.allclose(back, pts, atol=1e-6),
        f"max |dx| = {np.abs(back - pts).max():.2e}")
  check("a cloud has no faces", faces is None or len(faces) == 0)


def sphere_cloud(n=4000, radius=0.3, center=(0.5, 0.5, 0.5), seed=0):
  """Points on a sphere, normals the outward radial direction."""
  rng = np.random.default_rng(seed)
  v = rng.normal(size=(n, 3))
  v /= np.linalg.norm(v, axis=1, keepdims=True)
  return np.array(center) + radius * v, v


def check_sphere():
  print("\nsphere")
  if not os.path.exists(poisson.BINARY):
    check("binary present", False, f"{poisson.BINARY} -- run build_vendor.sh")
    return

  pts, normals = sphere_cloud()
  with tempfile.TemporaryDirectory() as d:
    verts, faces = poisson.reconstruct(pts, normals, d, depth=7)

  check("watertight", poisson.is_watertight(faces))
  check("Euler characteristic is 2",
        poisson.euler_characteristic(verts, faces) == 2,
        f"chi = {poisson.euler_characteristic(verts, faces)}")

  r = np.linalg.norm(verts - 0.5, axis=1)
  cell = 1 / 2 ** 7
  check("radius recovered", abs(r.mean() - 0.3) < 0.25 * cell,
        f"mean {r.mean():.5f} vs 0.3, spread {r.std():.5f}, cell {cell:.5f}")

  # Volume is the check the radius alone cannot make: a lumpy shell can average
  # to the right radius. 4/3 pi r^3 = 0.11310.
  expected = 4 / 3 * np.pi * 0.3 ** 3
  vol = mesh_volume(verts, faces)
  check("volume recovered", abs(vol - expected) / expected < 0.01,
        f"{vol:.5f} vs {expected:.5f}")


def check_normals():
  print("\nnormals")
  rng = np.random.default_rng(1)
  X = rng.random((400, 2))
  # A plane, which a local *linear* fit must reproduce exactly at any bandwidth.
  z = 0.3 + 0.8 * X[:, 0] - 0.5 * X[:, 1]

  value, grad = poisson.local_plane(X, z, X, h=0.15)
  check("value exact on a plane", np.allclose(value, z, atol=1e-9),
        f"max err {np.abs(value - z).max():.2e}")
  check("gradient exact on a plane",
        np.allclose(grad, [0.8, -0.5], atol=1e-9),
        f"mean {grad.mean(axis=0)}")

  n = poisson.upward_normals(grad, relief=1.0)
  truth = np.array([-0.8, 0.5, 1.0]) / np.linalg.norm([-0.8, 0.5, 1.0])
  check("normal is the plane's own", np.allclose(n, truth, atol=1e-9))
  check("normals are unit", np.allclose(np.linalg.norm(n, axis=1), 1.0))

  # relief has to enter *before* normalization, or a landscape scaled in z would
  # keep the normals of the unscaled one.
  flat = poisson.upward_normals(grad, relief=0.1)
  check("relief tilts the normal toward vertical",
        flat[:, 2].mean() > n[:, 2].mean(),
        f"nz {n[:, 2].mean():.3f} at relief 1 -> {flat[:, 2].mean():.3f} at 0.1")


def check_box():
  print("\nflat landscape")
  if not os.path.exists(poisson.BINARY):
    check("binary present", False, "run build_vendor.sh")
    return

  # A dense square of paragraphs all scoring 0.8. The solid under that landscape
  # is a slab, and its volume is one thing we can write down in advance.
  g = np.linspace(0.15, 0.85, 40)
  X = np.column_stack([a.ravel() for a in np.meshgrid(g, g)])
  z = np.full(len(X), 0.8)

  cloud, (gx, gy, mask) = poisson.build_cloud(
    X, z, relief=0.5, radius=0.05, resolution=120, bandwidth=0.06,
    wall_spacing=0.02, floor_stride=2)
  counts = cloud.counts()
  check("all three parts present", all(counts.values()), str(counts))
  check("lid is level",
        np.allclose(cloud.points[cloud.part == "lid"][:, 2], 0.4),
        "score 0.8 x relief 0.5")
  check("lid normals point straight up",
        np.allclose(cloud.normals[cloud.part == "lid"][:, 2], 1.0, atol=1e-6))
  check("wall normals are horizontal",
        np.allclose(cloud.normals[cloud.part == "wall"][:, 2], 0.0))

  # The check that caught the real bug. An inward-pointing wall normal does not
  # look wrong in a plot of the cloud -- it is still horizontal, still on the
  # rim -- and what it produces downstream is a mesh that is still a closed
  # surface, just not the boundary of this solid. On a footprint this convex,
  # outward means "away from the centroid" and every single one should be.
  wall = cloud.part == "wall"
  w = cloud.points[wall][:, :2]
  outward = np.einsum("ij,ij->i", cloud.normals[wall][:, :2], w - w.mean(axis=0))
  check("wall normals point outward", (outward > 0).all(),
        f"{(outward > 0).mean():.1%} of {wall.sum()}")

  with tempfile.TemporaryDirectory() as d:
    verts, faces = poisson.reconstruct(cloud.points, cloud.normals, d, depth=7)

  verts, faces, dropped = poisson.keep_largest(verts, faces)
  check("at most a negligible spurious component",
        all(n < 0.01 * len(faces) for n in dropped),
        f"dropped {dropped} triangles of {len(faces)}")
  check("watertight", poisson.is_watertight(faces))
  check("Euler characteristic is 2",
        poisson.euler_characteristic(verts, faces) == 2,
        f"chi = {poisson.euler_characteristic(verts, faces)}")

  # The footprint is the mask, not the 0.7 x 0.7 square of paragraphs: the
  # support radius grows it by ~one radius on every side. Take the area from
  # the mask itself and only the height from arithmetic.
  cell = (gx[0, 1] - gx[0, 0]) * (gy[1, 0] - gy[0, 0])
  expected = mask.sum() * cell * 0.4
  vol = mesh_volume(verts, faces)
  check("volume is footprint x height", abs(vol - expected) / expected < 0.05,
        f"{vol:.5f} vs {expected:.5f}")


def main():
  print(f"PoissonRecon binary: {poisson.BINARY}")
  check_ply_roundtrip()
  check_normals()
  check_sphere()
  check_box()

  print()
  if FAILURES:
    print(f"{len(FAILURES)} check(s) failed: {', '.join(FAILURES)}")
    return 1
  print("all checks passed")
  return 0


if __name__ == "__main__":
  sys.exit(main())
