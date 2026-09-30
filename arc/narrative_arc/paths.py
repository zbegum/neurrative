"""Where data is read from and outputs are written, and how a run records itself.

Outputs land under

    <output-dir>/<book>/<model>/windows/w40_s20/series.npz
    <output-dir>/<book>/<model>/arc_2d/<pca|umap|tsne>/...
    <output-dir>/<book>/<model>/arc_3d/<pca|umap|tsne>/...

Every parameter that changes the numbers (window, stride, projection settings,
fit) is spelled into the filename, so runs at different settings never
overwrite each other. `stamp()` writes a params.json beside the outputs with the
full argument list, so a figure can always be traced back to how it was made.
"""

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# The books live at the repository root, shared with the rest of neurrative.
DEFAULT_DATA_DIR = os.path.join(os.path.dirname(ROOT), "books")
DEFAULT_OUTPUT_DIR = os.path.join(ROOT, "output")

WINDOWS = "windows"
ARC_2D = "arc_2d"
ARC_3D = "arc_3d"


def window_tag(size, stride, l2=False):
  """`w40_s20`, or `w40_s20_l2` when embeddings were normalized before pooling."""
  return f"w{size}_s{stride}" + ("_l2" if l2 else "")


def out_dir(output_root, book, model, *parts, create=True):
  path = os.path.join(output_root, book, model, *parts)
  if create:
    os.makedirs(path, exist_ok=True)
  return path


def stamp(directory, script, args=None, **extra):
  """Write `params.json` beside the outputs: the repository-wide format.

  One implementation, in common/paths.py, so every run in the repository
  records itself the same way.
  """
  return _repo_paths().stamp(directory, script, args, **extra)


def _repo_paths():
  """common/paths.py, loaded by file so its name cannot clash with this module."""
  import importlib.util
  spec = importlib.util.spec_from_file_location(
    "neurrative_common_paths", os.path.join(os.path.dirname(ROOT), "common", "paths.py"))
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module
