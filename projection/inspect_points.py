"""
Backtrack points in a 2-D projection to their source paragraphs.

Row i of the coordinate array corresponds to paragraph i in processed.json,
so any dot on a plot maps directly to a paragraph's id and text.

Two modes:

  1. Outliers (default): the N points farthest from the centroid.

     python projection/inspect_points.py --book alice_wonderland --model bge-m3
     python projection/inspect_points.py --book alice_wonderland --model bge-m3 --proj umap

  2. Bounding box: every paragraph whose (x, y) falls in a region you read
     off the plot's axes.

     python projection/inspect_points.py --book alice_wonderland --model bge-m3 \
       --proj umap --xmin -7 --xmax -5 --ymin 8 --ymax 9
"""

import os
import glob
import json
import argparse
import sys

import numpy as np

# The repository root and common/ on the path, wherever this script lives.
_ROOT = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_ROOT, "common")):
  _ROOT = os.path.dirname(_ROOT)
sys.path[1:1] = [_ROOT, os.path.join(_ROOT, "common")]

import paths
from data import load_paragraphs


def load_coords(book_dir, model, proj):
  """Return the 2-D coordinates for the requested projection.

  PCA has a single canonical file. UMAP filenames are tagged with their
  parameters, so we fall back to the newest match if the bare name is absent.
  """
  # Repo-root relative, so this resolves the same whether it is run from the
  # repo root or from projection/ -- the paths it prints are quoted back to
  # the user, and a working-directory-dependent one would be misleading.
  out_dir = paths.out_dir(book_dir, model, paths.STRUCTURE_2D, create=False)

  if proj == "pca":
    path = paths.find_coords(out_dir, "pca.npy")
  else:
    path = paths.find_coords(out_dir, "umap.npy")
    if path is None:
      # Both layouts are searched: migrated directories keep these under umap/,
      # the ones that have not been re-run yet still have them flat.
      candidates = sorted(glob.glob(os.path.join(out_dir, "umap", "umap_*.npy")) +
                          glob.glob(os.path.join(out_dir, "umap_*.npy")))
      if not candidates:
        raise FileNotFoundError(
          f"No UMAP coordinates under {out_dir}. Run embedding.py first."
        )
      path = candidates[-1]

  if path is None or not os.path.exists(path):
    raise FileNotFoundError(
      f"Coordinates for {proj} not found under {out_dir}. Run embedding.py first.")

  print(f"Coordinates: {path}")
  return np.load(path)


def show(paragraphs, indices, coords, chars=240):
  for rank, i in enumerate(indices, 1):
    p = paragraphs[i]
    x, y = coords[i]
    text = p["text"]
    if len(text) > chars:
      text = text[:chars] + " ..."
    print(f"\n[{rank}] row {i} | {p['id']} | chapter {p['chapter_id']} "
          f"| ({x:+.2f}, {y:+.2f})")
    print(f"    {text}")


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--book", required=True)
  parser.add_argument("--model", required=True)
  parser.add_argument("--proj", default="pca", choices=["pca", "umap"])
  parser.add_argument("--top", type=int, default=10,
                      help="Number of farthest-from-center outliers to show.")
  # Optional bounding box (read off the plot axes). If any is set, box mode wins.
  parser.add_argument("--xmin", type=float, default=None)
  parser.add_argument("--xmax", type=float, default=None)
  parser.add_argument("--ymin", type=float, default=None)
  parser.add_argument("--ymax", type=float, default=None)
  args = parser.parse_args()

  paragraphs = load_paragraphs(args.book)

  coords = load_coords(args.book, args.model, args.proj)
  if len(coords) != len(paragraphs):
    raise ValueError(
      f"coords ({len(coords)}) and paragraphs ({len(paragraphs)}) mismatch."
    )

  box = any(v is not None for v in (args.xmin, args.xmax, args.ymin, args.ymax))

  if box:
    x, y = coords[:, 0], coords[:, 1]
    mask = np.ones(len(coords), dtype=bool)
    if args.xmin is not None: mask &= x >= args.xmin
    if args.xmax is not None: mask &= x <= args.xmax
    if args.ymin is not None: mask &= y >= args.ymin
    if args.ymax is not None: mask &= y <= args.ymax
    indices = np.where(mask)[0]
    print(f"\n{len(indices)} paragraph(s) inside the box "
          f"x[{args.xmin}, {args.xmax}] y[{args.ymin}, {args.ymax}]:")
    show(paragraphs, indices, coords)
  else:
    center = coords.mean(axis=0)
    dist = np.linalg.norm(coords - center, axis=1)
    indices = np.argsort(dist)[::-1][:args.top]
    print(f"\nTop {args.top} outliers (farthest from centroid) in {args.proj.upper()}:")
    show(paragraphs, indices, coords)


if __name__ == "__main__":
  main()
