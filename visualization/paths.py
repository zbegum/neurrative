"""One place where output paths are decided, and where a run records itself.

Two problems this fixes.

**Silent overwrite.** Several scripts wrote a fixed filename -- `emotion_axis_3d.png`,
`arc_on_surface_3d.png`, `bandwidth_wonder.png` -- regardless of the bandwidth,
window or temperature they were run at. Re-running with different settings
replaced the previous figure, so the PNG on disk could not be attributed to any
particular set of parameters. `variant()` turns the parameters that change the
result into a directory name, so two settings produce two directories instead of
one file twice.

**Two bandwidth conventions, one filename.** `emotion_surface.py` (isotropic `h`
in raw PCA units, chosen by leave-one-out CV) and `smooth_gaussian_nw.py`
(`hx, hy` in standardized units, chosen by k-fold CV) run the *same* estimator --
`geometry.smoothers.nadaraya_watson` -- but tune it differently, and both save
`field_<emotion>.npz`. The surfaces are therefore close but not equal, and the
bandwidths are not even in comparable units. `stamp()` writes a `params.json`
next to every artifact naming the estimator, its parameters and the script that
ran, so a `.npz` can always say which convention produced it.

Usage:

    import paths
    out = paths.out_dir(args.book, args.model,
                        os.path.join(paths.NARRATIVE_ARC_3D,
                                     paths.ARC_FROM_TERRAIN),
                        {"w": args.size, "s": args.stride,
                         "h": args.hx})
    paths.stamp(out, __file__, args, method={"smoother": "gaussian_nw",
                                             "hx": args.hx, "hy": args.hy})

`variant` is for the knobs that change the numbers. Cosmetic flags (`--resolution`,
colormaps) belong in `params.json` only -- they land there automatically, because
`stamp` records the whole argparse namespace.
"""

import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The canonical subdirectory per figure family. Scripts import these names rather
# than repeating a string literal: five scripts used to hardcode "arc_on_surface"
# inline, so renaming the directory would have broken them silently.
STRUCTURE_2D = "structure_2d"
PROJECTION_COMPARISON = "projection_comparison"
EMOTIONS_3D = "emotions_3d"

# UMAP run over a kNN graph with reading-order edges added, swept over how much
# those edges are worth. Its own family rather than another projection under
# `structure_2d`: the output is a sweep with a Pareto curve attached, not a
# layout at settings you chose, and `structure_2d`'s per-projection folders are
# organised around one file per (method, parameters) with no metrics beside it.
CHAIN_UMAP = "chain_umap"

# Everything that fits z = f(PC1, PC2) lives under one parent. It is named for
# the output, not the operation: nothing is "smoothed" -- the scores are never
# modified -- a surface is fitted to them. The subfolders below are its parts.
SURFACE = "surface"
SURFACE_RAW = "raw"                  # the unsmoothed points, for comparison
SURFACE_ISOTROPIC = "isotropic_loo"  # one h, leave-one-out, raw PCA units
SURFACE_SWEEP = "bandwidth_sweep"    # method x bandwidth, under- to over-smoothed
# The only fit in this family that is not a kernel smoother: a least-squares
# tensor-product B-spline (`b-surface/`). Its knot count, not a bandwidth, is
# what decides the smoothing, so its variant directory is `k<n>` and it is a
# sibling rather than another entry in the bandwidth sweep.
SURFACE_BSPLINE = "bspline_ls"
# The same B-spline fit carried over the whole unit square instead of stopping
# at the paragraphs. Its own directory rather than a variant of bspline_ls,
# because the *coordinates* differ: PC1 and PC2 are rescaled into [0, 1], so its
# gx/gy are not in pca.npy units and its figures cannot be laid over a sibling's
# without the transform. Every artifact carries that transform, and the filename
# says `pca_unit` rather than `pca` for the same reason.
SURFACE_BSPLINE_UNIT = "bspline_ls_unit"
# The only fit in this family that is not a height field at all: screened
# Poisson reconstruction (`closed-surface/`) returns the watertight boundary of
# the solid under the landscape, so it writes a mesh where its siblings write
# field_<emotion>_pca.npz. It sits here anyway -- it is the same (PC1, PC2,
# score) cloud, fitted -- and its variant directory is d<depth> because the
# octree depth is what plays the bandwidth's part.
SURFACE_POISSON = "poisson_closed"
# The same reconstruction with the lid cut off it and the footprint widened to
# the whole PCA rectangle: an *open* surface, so this one does write
# field_<emotion>_pca.npz and is a drop-in for the kernel fits. Its own
# directory rather than a variant of poisson_closed, because the file contract
# differs -- a mesh there, a field here.
SURFACE_POISSON_OPEN = "poisson_open"
# The four anisotropic estimators get a subfolder named after themselves
# (gaussian_nw, epanechnikov_nw, local_linear, loess): hx/hy tuned by k-fold CV
# on standardized coordinates. `isotropic_loo` runs the *same* estimator under a
# different bandwidth rule, which is why it is a sibling rather than folded in --
# both write field_<emotion>.npz, and the parent directory is what tells them
# apart.

# The base plane a fitted surface lives over. PCA is the only projection that can
# carry one: it is linear and metric, so every paragraph has one well-defined
# (x, y) and "the height here" means something. UMAP and t-SNE are non-metric --
# two paragraphs can land on the same spot with different scores -- so z = f(x, y)
# is not a function there. Constant for now, but named rather than assumed, and
# stamped into every filename so a figure pulled out of its directory still says
# which plane it was fitted over.
BASE_PROJECTION = "pca"

GEODESICS = "geodesics"

# The windowed series itself -- pooled vectors, window ranges, pooled scores --
# before any projection has been chosen. Everything under `narrative_arc*` and
# `arc_*` is a drawing of what lands here, so this is the one directory in the
# family that holds data rather than a picture. Written by `windows.py`; its
# variant is the window setting, `w40_s20`.
WINDOWS = "windows"

NARRATIVE_ARC = "narrative_arc"
ARC_COMPARISON = "arc_comparison"

# The arc in 3-D, all of it. Seven scripts used to split across `narrative_arc_3d`
# and `arc_on_surface` as though they were different objects; they are the same
# curve with the third axis derived differently. The subfolder says how, which is
# the only thing that actually distinguishes them:
#
#   height_from_text      z is the emotion the window itself carried, smoothed
#                         over reading position. Height comes from *when*.
#   height_from_terrain   z is read off the fitted emotion surface at the arc's
#                         (x, y). Height comes from *where*.
#   mood_axis             the six emotions collapsed to one interpretable height.
#   curve_smoothing       the arc itself smoothed on the surface (Pawellek 2024).
NARRATIVE_ARC_3D = "narrative_arc_3d"
ARC_FROM_TEXT = "height_from_text"
ARC_FROM_TERRAIN = "height_from_terrain"
ARC_MOOD_AXIS = "mood_axis"
ARC_CURVE_SMOOTHING = "curve_smoothing"

# Inside mood_axis: the mood collapsed **per paragraph** first, then one Gaussian
# surface fitted to those 789 numbers. `arc_emotion_axis.py` does it the other way
# round -- six surfaces, blended per grid cell -- so the two live apart.
MOOD_ON_POINTS = "gaussian_on_points"

# Results produced by a script that no longer exists, or by a method the notes
# have since superseded. Kept (they are real runs) but out of the way, so the
# live directories only hold things the current code can reproduce.
DEPRECATED = "_deprecated"

# Inside `structure_2d`, one subdirectory per projection. 54 files in one flat
# directory is three unrelated sweeps interleaved by alphabetical accident;
# grouped by method, each folder is one method's coordinates, single views and
# parameter sweep together.
#
# `emotions_3d` tried per-projection subdirectories and they were quarantined --
# but read `_deprecated/README.md` for why: the failure was that the *flat and
# nested layouts coexisted*, so the same figure appeared twice under two paths.
# The layout was not the bug, the duplication was. `migrate_projections` below is
# how that is not repeated: it moves rather than copies, and runs before every
# write, so a directory is never in both layouts at once.
PROJECTIONS = ("pca", "umap", "tsne")

# Which projection owns a file, by directory family. `storyline` is a UMAP
# product under a name that does not say so, and `structure_2d`'s sweep grids are
# per-method so they fold in with their method.
_STRUCTURE_OWNERS = (
  ("pca", "pca"),
  ("umap", "umap"),
  ("storyline", "umap"),
  ("grid_umap", "umap"),
  ("tsne", "tsne"),
  ("grid_tsne", "tsne"),
)

# `narrative_arc` spells t-SNE with the hyphen (`method.lower()` of "t-SNE"), and
# prefixes its figures with `arc_` while its coordinates carry the bare method.
# Its `_grid_*` files match nothing here on purpose: they are cross-method
# summaries and stay at the top of the directory rather than folding into any one
# method's folder.
_ARC_OWNERS = (
  ("arc_pca", "pca"),
  ("arc_umap", "umap"),
  ("arc_t-sne", "tsne"),
  ("pca", "pca"),
  ("umap", "umap"),
  ("t-sne", "tsne"),
)


def _owner_of(filename, owners):
  for prefix, owner in owners:
    if filename.startswith(prefix):
      return owner
  return None


def projection_of(filename):
  """The projection a `structure_2d` file belongs to, or None.

  None means the file is not a projection artifact -- `README.md`,
  `params.json` -- and belongs at the top of `structure_2d`, not in a method
  folder.
  """
  return _owner_of(filename, _STRUCTURE_OWNERS)


def arc_projection_of(filename):
  """The projection a `narrative_arc` file belongs to, or None.

  None keeps a file at the top of the directory. That is what happens to the
  `_grid_*` sweeps, which compare the projections against each other and so
  belong to none of them.
  """
  return _owner_of(filename, _ARC_OWNERS)


def projection_dir(structure_dir, projection, create=True):
  """`structure_2d/<projection>/`."""
  if projection not in PROJECTIONS:
    raise ValueError(f"Unknown projection {projection!r}; expected one of "
                     f"{PROJECTIONS}.")
  path = os.path.join(structure_dir, projection)
  if create:
    os.makedirs(path, exist_ok=True)
  return path


def migrate_projections(structure_dir, owner=projection_of):
  """Move any flat projection files in `structure_dir` into their method folder.

  Idempotent, and the reason both layouts can never coexist: a script that is
  about to write calls this first, so whatever an earlier version left flat is
  folded in rather than shadowed. Returns the number of files moved.

  Directories that have never been migrated therefore cost one run, not a
  reorganization -- which is why only the pair that needed it was moved by hand.

  `owner` decides which method a filename belongs to, since the families spell
  their methods differently: `projection_of` for `structure_2d`,
  `arc_projection_of` for `narrative_arc`. Whatever it maps to None stays put.
  """
  if not os.path.isdir(structure_dir):
    return 0

  moved = 0
  for name in sorted(os.listdir(structure_dir)):
    src = os.path.join(structure_dir, name)
    if not os.path.isfile(src):
      continue
    projection = owner(name)
    if projection is None:
      continue
    dst = os.path.join(projection_dir(structure_dir, projection), name)
    if os.path.exists(dst):
      # Same name in both layouts: the nested one is canonical, and the flat one
      # is what an older run left behind. Dropping it is the whole point.
      os.remove(src)
    else:
      os.rename(src, dst)
    moved += 1
  return moved


def find_coords(structure_dir, filename, projection=None, owner=projection_of):
  """Locate saved projection coordinates, nested layout first, flat second.

  Readers go through this so that a directory which has not been migrated yet
  still resolves. Returns None if neither layout has the file, leaving the
  caller to raise the error it wants.
  """
  projection = projection or owner(filename)
  if projection:
    nested = os.path.join(structure_dir, projection, filename)
    if os.path.exists(nested):
      return nested

  flat = os.path.join(structure_dir, filename)
  return flat if os.path.exists(flat) else None


def _fmt(v):
  """Compact, filename-safe rendering of a parameter value.

  Floats lose their trailing zeros (0.150 -> 0.15) so that the same bandwidth
  always spells the same directory, whether it arrived as 0.15 or 0.150.
  """
  if isinstance(v, bool):
    return "on" if v else "off"
  if isinstance(v, float):
    return f"{v:g}"
  if isinstance(v, (list, tuple)):
    return "-".join(_fmt(x) for x in v)
  return str(v).replace(os.sep, "_").replace(" ", "")


def variant(params):
  """A directory name from the parameters that change the result.

  `{"w": 10, "s": 5, "h": 0.15}` -> `"w10_s5_h0.15"`. Order is the caller's, so
  keep it stable across runs of the same script. `None` values are dropped, which
  lets a caller pass an optional knob without branching.
  """
  parts = [f"{k}{_fmt(v)}" for k, v in params.items() if v is not None]
  return "_".join(parts) if parts else "default"


def named(stem, emotion, ext="png", projection=BASE_PROJECTION):
  """A figure filename that says which projection it was fitted over.

  `named("surface", "wonder")` -> `surface_wonder_pca.png`.

  The projection trails, like every other run parameter in this repo's filenames
  (`w10_s5`, `n15_d0.1_s0`), so the stem still sorts the way it reads. Directories
  that are projection-agnostic -- `structure_2d`, `narrative_arc` -- already name
  the method themselves and do not use this.
  """
  return f"{stem}_{emotion}_{projection}.{ext.lstrip('.')}"


def out_dir(book, model, figure, variant_params=None, create=True):
  """`output/<book>/<model>/<figure>[/<variant>]`, created by default."""
  parts = [ROOT, "output", book, model, figure]
  if variant_params:
    parts.append(variant(variant_params))
  path = os.path.join(*parts)
  if create:
    os.makedirs(path, exist_ok=True)
  return path


def _git_commit():
  """The commit the code was at, or None outside a git checkout.

  Most of this repo's analysis code is untracked, so this is a weak provenance
  signal on its own -- it is recorded because it costs nothing when it does work.
  """
  try:
    out = subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True, timeout=5)
    return out.stdout.strip() or None
  except (OSError, subprocess.SubprocessError):
    return None


def stamp(directory, script, args=None, **extra):
  """Write `params.json` describing the run that filled this directory.

  `script` is normally `__file__`. `args` is the argparse namespace -- every flag
  it holds is recorded, including the cosmetic ones left out of the variant name.
  `extra` is for anything argparse does not know: the estimator actually used, a
  CV-selected bandwidth, the emotions resolved from the scores file.

  Returns the path written, so a caller can log it.
  """
  payload = {
    "script": os.path.basename(script),
    "command": " ".join([os.path.basename(sys.argv[0])] + sys.argv[1:]),
    "git_commit": _git_commit(),
    "args": {k: v for k, v in sorted(vars(args).items())} if args else {},
  }
  payload.update(extra)

  path = os.path.join(directory, "params.json")
  with open(path, "w") as f:
    json.dump(payload, f, indent=2, default=str)
    f.write("\n")
  return path
