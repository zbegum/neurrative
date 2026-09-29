"""Reconstruct one closed emotion landscape per emotion, for one book.

  python closed-surface/fit_surface.py --book alice_wonderland --model bge-m3
  python closed-surface/fit_surface.py --book alice_wonderland --model bge-m3 \
    --emotions wonder --depth 9 --relief 0.6

(x, y) is the PCA plane, z is the emotion score, and what comes out is the
watertight boundary of the solid under that landscape -- see `poisson.py` for
why a closed solid rather than a height field, and what the floor and walls are
doing in the point cloud.

Output goes to `output/<book>/<model>/surface/poisson_closed/<variant>/`,
beside `bspline_ls` and the kernel smoothers, because it is another fit of the
same object. It does *not* write the `field_<emotion>_pca.npz` those write: a
closed surface has no `Z` on a grid to write, and faking one by keeping only the
upper sheet would hand the geodesic stack a height field that had been through a
solid reconstruction for no reason. What it writes is a mesh.
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "visualization"))

import poisson
import paths
from data import load_paragraphs, load_scores
from smooth_common import load_pca

CMAP = "viridis"


def normalize(X):
  """PCA coordinates into [0, 1], one scale for both axes.

  Both axes share a scale so the PCA plane keeps its aspect ratio: PC1 carries
  more variance than PC2 and the cloud is genuinely wider than it is tall.
  Squaring it into a unit square would be a claim about the data. Returns the
  scaled points and the (offset, scale) needed to read a length back in PCA
  units.
  """
  X = np.asarray(X, float)
  lo = X.min(axis=0)
  scale = float((X.max(axis=0) - lo).max())
  return (X - lo) / scale, (lo, scale)


def nn_radius(X, percentile=99.0):
  """The support radius, read off the cloud's own nearest-neighbour spacing.

  Same rule and same reasoning as `b-surface/bspline_surface.py`: a point of the
  plane further than this from every paragraph is outside the book, not between
  two of its paragraphs. Duplicated rather than imported because there it sets
  what is *drawn* and here it sets what is *built* -- the footprint of the
  solid -- and the two should be free to diverge.
  """
  d = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
  np.fill_diagonal(d, np.inf)
  return float(np.percentile(d.min(axis=1), percentile))


def mesh_plot(verts, faces, cloud, emotion, subtitle, output_path,
              elev=28, azim=-60):
  fig = plt.figure(figsize=(11, 8))
  ax = fig.add_subplot(111, projection="3d")

  ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2],
                  cmap=CMAP, linewidth=0.0, antialiased=True, alpha=0.9,
                  shade=True)
  lid = cloud.points[cloud.part == "lid"]
  ax.scatter(lid[:, 0], lid[:, 1], lid[:, 2], c="#33322e", s=3, alpha=0.4,
             depthshade=False)

  ax.set_xlabel("PC1 (unit box)")
  ax.set_ylabel("PC2 (unit box)")
  ax.set_zlabel(emotion)
  ax.view_init(elev=elev, azim=azim)
  ax.set_title(f"{emotion}: closed landscape over the PCA map\n{subtitle}")

  fig.tight_layout()
  fig.savefig(output_path, dpi=200)
  plt.close(fig)
  print(f"  wrote {output_path}")


def support_outline(gx, gy, support, height, relief):
  """The support boundary as 3-D polylines lying on the surface.

  Drawn on the landscape rather than projected onto the floor of the plot,
  because the question it answers -- "is this ridge made of paragraphs?" -- is
  asked about a feature at its own height, and a line on the floor of a 3-D
  axes is at the wrong end of the reader's eye.
  """
  fig, ax = plt.subplots()
  segments = ax.contour(gx, gy, support.astype(float), levels=[0.5]).allsegs[0]
  plt.close(fig)

  dx, dy = gx[0, 1] - gx[0, 0], gy[1, 0] - gy[0, 0]
  out = []
  for seg in segments:
    ix = np.clip(((seg[:, 0] - gx[0, 0]) / dx).round().astype(int),
                 0, gx.shape[1] - 1)
    iy = np.clip(((seg[:, 1] - gy[0, 0]) / dy).round().astype(int),
                 0, gy.shape[0] - 1)
    # A hair above the surface, or the line disappears into it under z-buffer
    # ties on the flat parts.
    out.append((seg[:, 0], seg[:, 1], relief * height[iy, ix] + 0.004))
  return out


def surface_plot(gx, gy, Z, X, z, support_contour, emotion, subtitle,
                 output_path, elev=28, azim=-60):
  """The open surface, with the edge of the support drawn on it.

  The line matters more here than in any other figure this repo draws: inside it
  the surface is fitted to paragraphs, outside it the surface is an extension of
  that fit over a part of the plane the book never visits. They are the same
  colour and the same smoothness, and only the line says which is which.
  """
  fig = plt.figure(figsize=(11, 8))
  ax = fig.add_subplot(111, projection="3d")

  ax.plot_surface(gx, gy, Z, cmap=CMAP, linewidth=0, antialiased=True,
                  alpha=0.9, rstride=2, cstride=2)
  ax.scatter(X[:, 0], X[:, 1], z, c="#33322e", s=3, alpha=0.45,
             depthshade=False)

  for x, y, zc in support_outline(*support_contour):
    ax.plot(x, y, zc, color="#c46a3f", linewidth=1.8, zorder=10)

  ax.set_xlabel("PC1 (unit box)")
  ax.set_ylabel("PC2 (unit box)")
  ax.set_zlabel(emotion)
  ax.view_init(elev=elev, azim=azim)
  ax.set_title(f"{emotion}: open landscape over the whole PCA plane\n{subtitle}")

  fig.tight_layout()
  fig.savefig(output_path, dpi=200)
  plt.close(fig)
  print(f"  wrote {output_path}")


def open_sheet(verts, faces, cloud, gx, gy, X, z, emotion, offset, scale,
               output_dir, args):
  """Cut the lid off the solid, save it as a surface and as a field.

  The mesh this writes is *open*: it has a boundary, at the rim of the PCA
  rectangle, and `is_watertight` on it is False by design. That is the whole
  difference from the closed fit -- what is saved is the landscape alone,
  covering the plane, with the floor and walls that made the solve well posed
  cut away again.
  """
  sheet_v, sheet_f, shelves = poisson.top_sheet(verts, faces, args.min_nz)
  folded = poisson.folded_fraction(sheet_v, gx, gy)
  print(f"  sheet {len(sheet_v)} vertices, {len(sheet_f)} triangles "
        f"({len(sheet_f) / len(faces):.0%} of the solid's skin) | "
        f"overhanging cells {folded:.2%}")

  Zr, empty = poisson.upper_envelope(verts, faces, gx, gy)
  print(f"  envelope over {gx.shape[0]}x{gx.shape[1]} cells, {empty:.2%} of "
        f"them uncovered by any triangle and filled from a neighbour")

  # The outermost ring is where the lid runs into the wall that closed the
  # solid, and the seam between the two is the one place the reconstruction is
  # reliably wrong: a lid sample with a vertical normal and a wall sample with a
  # horizontal one, in the same cell, tear into a slot a few cells deep. Every
  # such cell in Alice sits within three of the border and none in the interior,
  # so the field is reported from just inside the seam. What is cut is padding
  # -- the grid is built past the data's bounding rectangle for this reason --
  # not a part of the plane any paragraph occupies.
  m = args.rim_margin
  if m:
    gx, gy, Zr = gx[m:-m, m:-m], gy[m:-m, m:-m], Zr[m:-m, m:-m]
    support = cloud.support[m:-m, m:-m]
    height = cloud.height[m:-m, m:-m]
    print(f"  trimmed {m} cells of scaffolding seam from each edge; the field "
          f"spans PC1 {gx.min():.3f}..{gx.max():.3f} of the unit box")
  else:
    support, height = cloud.support, cloud.height

  # Back to the units the sibling fits use: the original PCA plane, and a score
  # rather than a relief height. A field written in unit-box coordinates would
  # be a different object from every other field_<emotion>_pca.npz on disk.
  Z = Zr / args.relief
  gx_pca, gy_pca = gx * scale + offset[0], gy * scale + offset[1]
  print(f"  relief {Z.min():.2f}..{Z.max():.2f} in score units "
        f"({(Z < 0).mean():.1%} below 0, {(Z > 1).mean():.1%} above 1)")

  env_v, env_f = poisson.grid_mesh(gx, gy, Zr)
  np.savez(
    os.path.join(output_dir, paths.named("field", emotion, "npz")),
    gx=gx_pca, gy=gy_pca, Z=Z, mask=support, X=X * scale + offset,
    y=z, emotion=emotion, verts=env_v, faces=env_f,
    sheet_verts=sheet_v, sheet_faces=sheet_f,
    relief=args.relief, depth=args.depth, point_weight=args.point_weight,
    folded_fraction=folded,
  )

  surface_plot(gx, gy, Zr, X, args.relief * z,
               (gx, gy, support, height, args.relief), emotion,
               f"screened Poisson reconstruction, depth {args.depth}, "
               f"point weight {args.point_weight:g}, relief {args.relief:g}; "
               f"orange line = edge of the book's support",
               os.path.join(output_dir, paths.named("surface", emotion)))

  return {
    "sheet_vertices": len(sheet_v), "sheet_triangles": len(sheet_f),
    "dropped_shelves": shelves, "folded_fraction": folded,
    "envelope_interpolated": empty,
    "field_range": [float(Z.min()), float(Z.max())],
    "field_outside_unit": float(((Z < 0) | (Z > 1)).mean()),
  }


def cloud_plot(cloud, emotion, output_path):
  """The input, colored by which part of the boundary it is.

  Worth drawing every time: a reconstruction that looks wrong is almost always
  a cloud that was wrong, and this is the picture that says so.
  """
  fig = plt.figure(figsize=(9, 7))
  ax = fig.add_subplot(111, projection="3d")

  colors = {"lid": "#2b6ea3", "floor": "#b0aca4", "wall": "#c46a3f"}
  for part, color in colors.items():
    p = cloud.points[cloud.part == part]
    if len(p):
      ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=2, c=color, label=part,
                 alpha=0.6, depthshade=False)

  lid = cloud.part == "lid"
  step = max(1, int(lid.sum() // 120))
  q = cloud.points[lid][::step]
  n = cloud.normals[lid][::step]
  ax.quiver(q[:, 0], q[:, 1], q[:, 2], n[:, 0], n[:, 1], n[:, 2],
            length=0.06, color="#33322e", linewidth=0.6, alpha=0.8)

  ax.set_xlabel("PC1")
  ax.set_ylabel("PC2")
  ax.set_zlabel(emotion)
  ax.legend(loc="upper left", fontsize=8)
  ax.set_title(f"{emotion}: the oriented cloud handed to PoissonRecon")

  fig.tight_layout()
  fig.savefig(output_path, dpi=200)
  plt.close(fig)
  print(f"  wrote {output_path}")


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--book", default="alice_wonderland", type=str)
  parser.add_argument("--model", default="bge-m3", type=str)
  parser.add_argument("--emotions", nargs="*", default=None,
                      help="Subset of emotions (default: all).")
  parser.add_argument("--depth", default=8, type=int,
                      help="Octree depth. The smoothing knob, and it runs like "
                           "the knot count rather than a bandwidth: deeper is "
                           "less smoothing. 8 is 256 cells across the box.")
  parser.add_argument("--point-weight", default=4.0, type=float,
                      help="Screening weight. 0 is the unscreened 2006 solve, "
                           "which pulls away from the samples; higher holds the "
                           "surface to them.")
  parser.add_argument("--samples-per-node", default=1.5, type=float)
  parser.add_argument("--relief", default=0.5, type=float,
                      help="Height of a score of 1.0, in units of the PCA "
                           "plane's width. Not cosmetic: the octree is "
                           "isotropic, so this sets how much resolution the "
                           "vertical structure gets.")
  parser.add_argument("--bandwidth", default=0.06, type=float,
                      help="Local-plane bandwidth for the lid normals, in unit-"
                           "box units. Too small and the normals follow the "
                           "score noise; too large and the lid flattens.")
  parser.add_argument("--resolution", default=160, type=int,
                      help="Support-mask grid, which sets the floor and wall "
                           "sampling density.")
  parser.add_argument("--floor-stride", default=2, type=int,
                      help="Keep every nth mask cell for the floor. The floor "
                           "is scaffolding; it should not out-number the text.")
  parser.add_argument("--base", default=0.08, type=float,
                      help="Thickness of the plinth under z = 0, in unit-box "
                           "units. Keeps the solid from pinching where a score "
                           "is near zero -- half of Alice's sadness scores are. "
                           "0 removes it, and the topology report will say so.")
  parser.add_argument("--wall-spacing", default=0.02, type=float,
                      help="Vertical spacing of wall samples, in unit-box "
                           "units.")
  parser.add_argument("--open", action="store_true",
                      help="Fit over the whole PCA rectangle and save the lid "
                           "alone: an open surface with a boundary, plus the "
                           "field_<emotion>_pca.npz the sibling fits write. "
                           "Without it, the closed solid is what is saved.")
  parser.add_argument("--rim-margin", default=4, type=int,
                      help="With --open, cells trimmed from each edge of the "
                           "field: the seam where the lid meets the scaffolding "
                           "wall. 0 keeps it, slots and all.")
  parser.add_argument("--lid", default="fitted", choices=("fitted", "scores"),
                      help="Height of the paragraph samples: the local plane "
                           "fit's value (smooth, --bandwidth is the knob) or "
                           "the raw score (interpolated, and combed by the "
                           "disagreement between neighbouring paragraphs).")
  parser.add_argument("--merge-radius", default=0.004, type=float,
                      help="Paragraphs closer than this in the unit box are one "
                           "lid point, with the mean of their scores. PCA is "
                           "not injective, and two heights over one (x, y) "
                           "reconstruct as a chimney through the solid. 0 "
                           "keeps them, and the topology report will say so.")
  parser.add_argument("--fill-buffer", default=0.5, type=float,
                      help="With --open, how far the extension samples are held "
                           "back from the support, in support radii. 0 puts "
                           "them right against the paragraphs, which tears the "
                           "solid along the boundary.")
  parser.add_argument("--min-nz", default=0.1, type=float,
                      help="With --open, how upward a triangle's normal must "
                           "point to be part of the lid rather than the walls. "
                           "0.1 keeps anything under 84 degrees from level.")
  parser.add_argument("--support-percentile", default=99.0, type=float)
  parser.add_argument("--verbose", action="store_true",
                      help="Echo PoissonRecon's own output.")
  args = parser.parse_args()

  paragraphs = load_paragraphs(args.book)
  emotions, matrix = load_scores(args.book, paragraphs)
  X_pca = load_pca(args.book, args.model)
  if len(X_pca) != len(paragraphs):
    raise ValueError(f"pca ({len(X_pca)}) and paragraphs ({len(paragraphs)}) "
                     f"mismatch.")

  if args.emotions:
    unknown = [e for e in args.emotions if e not in emotions]
    if unknown:
      raise ValueError(f"Unknown emotion(s) {unknown}. Available: {emotions}")
    selected = args.emotions
  else:
    selected = emotions

  X, (offset, scale) = normalize(X_pca)
  radius = nn_radius(X, args.support_percentile)
  print(f"{len(X)} paragraphs | PCA plane scaled by 1/{scale:.4f} into "
        f"[0,1]^2 | support radius {radius:.4f}")

  output_dir = paths.out_dir(
    args.book, args.model,
    os.path.join(paths.SURFACE,
                 paths.SURFACE_POISSON_OPEN if args.open
                 else paths.SURFACE_POISSON),
    {"d": args.depth, "pw": args.point_weight, "r": args.relief,
     "b": args.base},
  )

  report = {}
  for emotion in selected:
    z = matrix[:, emotions.index(emotion)]
    print(f"\n{emotion}:")

    cloud, (gx, gy, mask) = poisson.build_cloud(
      X, z, args.relief, radius, args.resolution, args.bandwidth,
      args.wall_spacing, args.floor_stride, args.base,
      footprint="box" if args.open else "support",
      fill_buffer=args.fill_buffer, merge_radius=args.merge_radius,
      lid=args.lid)
    counts = cloud.counts()
    if cloud.merged:
      print(f"  merged {cloud.merged} paragraph(s) onto coincident neighbours "
            f"({cloud.merged / len(X):.1%} of the book)")
    print(f"  cloud {len(cloud.points)} oriented points "
          f"({counts['lid']} lid, {counts['floor']} floor, "
          f"{counts['wall']} wall) over {mask.mean():.0%} of the bounding box")
    if args.open:
      print(f"  lid: {len(X)} paragraphs + {counts['lid'] - len(X)} extension "
            f"samples over the {1 - cloud.support.mean():.0%} of the rectangle "
            f"the book does not occupy")

    work = os.path.join(output_dir, f"work_{emotion}")
    os.makedirs(work, exist_ok=True)
    verts, faces = poisson.reconstruct(
      cloud.points, cloud.normals, work, depth=args.depth,
      point_weight=args.point_weight,
      samples_per_node=args.samples_per_node, verbose=args.verbose)

    verts, faces, dropped = poisson.keep_largest(verts, faces)
    if dropped:
      print(f"  dropped {len(dropped)} spurious component(s), "
            f"{sum(dropped)} triangles ({sum(dropped) / len(faces):.2%})")

    chi = poisson.euler_characteristic(verts, faces)
    tight = poisson.is_watertight(faces)
    resid = poisson.lid_residual(verts, cloud.points[cloud.part == "lid"])
    print(f"  mesh {len(verts)} vertices, {len(faces)} triangles | "
          f"watertight {tight} | Euler characteristic {chi}")
    print(f"  paragraph-to-nearest-vertex distance: median {np.median(resid):.4f}"
          f", 95th {np.percentile(resid, 95):.4f}, max {resid.max():.4f} "
          f"(cell size {1 / 2 ** args.depth:.4f})")

    np.savez(
      os.path.join(output_dir, paths.named("mesh", emotion, "npz")),
      verts=verts, faces=faces, emotion=emotion,
      cloud_points=cloud.points, cloud_normals=cloud.normals,
      cloud_part=cloud.part, mask=mask, gx=gx, gy=gy,
      X=X, y=z, pca_offset=offset, pca_scale=scale, relief=args.relief,
      depth=args.depth, point_weight=args.point_weight,
    )

    entry = {
      "cloud": counts,
      "vertices": len(verts), "triangles": len(faces),
      "watertight": bool(tight), "euler_characteristic": int(chi),
      "dropped_components": dropped,
      "lid_residual_median": float(np.median(resid)),
      "lid_residual_p95": float(np.percentile(resid, 95)),
    }

    if args.open:
      entry.update(open_sheet(verts, faces, cloud, gx, gy, X, z, emotion,
                              offset, scale, output_dir, args))
    else:
      mesh_plot(verts, faces, cloud, emotion,
                f"screened Poisson reconstruction, depth {args.depth}, "
                f"point weight {args.point_weight:g}, relief {args.relief:g}",
                os.path.join(output_dir, paths.named("mesh", emotion)))

    cloud_plot(cloud, emotion,
               os.path.join(output_dir, paths.named("cloud", emotion)))
    report[emotion] = entry

  paths.stamp(output_dir, __file__, args,
              stack="closed-surface.poisson",
              estimator="screened Poisson surface reconstruction "
                        "(vendor: mkazhdan/PoissonRecon)",
              emotions=selected, support_radius=radius, pca_scale=scale,
              results=report)

  print(f"\nDone.\n{output_dir}")


if __name__ == "__main__":
  main()
