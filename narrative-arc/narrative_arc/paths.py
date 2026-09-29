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

import json
import os
import subprocess
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_DIR = os.path.join(ROOT, "data")
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


def _git_commit():
  try:
    return subprocess.run(
      ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
      capture_output=True, text=True, check=True,
    ).stdout.strip() or None
  except (OSError, subprocess.CalledProcessError):
    return None


def stamp(directory, script, args=None, **extra):
  """Write `params.json`: the script, its arguments, the commit, and `extra`."""
  record = {
    "script": os.path.relpath(os.path.abspath(script), ROOT),
    "written": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "commit": _git_commit(),
  }
  if args is not None:
    record["args"] = vars(args)
  record.update(extra)
  with open(os.path.join(directory, "params.json"), "w") as f:
    json.dump(record, f, indent=2, default=str)
