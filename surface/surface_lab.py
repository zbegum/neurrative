"""
The surface lab: every saved surface for one book and emotion, side by side in
3-D, rotating together, on one shared height and colour scale.

Panels: the raw paragraph cloud, the four kernel smoothers, the leave-one-out
kernel, the least-squares B-spline and the open Poisson surface -- whichever of
them have been run for the book. Drag any panel to turn all of them.

Example:

python surface/surface_lab.py
"""

import argparse
import os
import sys

import numpy as np

# The repository root and common/ on the path, wherever this script lives.
_ROOT = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_ROOT, "common")):
  _ROOT = os.path.dirname(_ROOT)
sys.path[1:1] = [_ROOT, os.path.join(_ROOT, "common")]

import viewer as V

HERE = os.path.dirname(os.path.abspath(__file__))
GRID = 80
# (name on the page, figure family of the saved surface), in panel order
METHODS = [("gaussian", "surface/gaussian_nw"), ("epanechnikov", "surface/epanechnikov_nw"),
           ("local linear", "surface/local_linear"), ("loess", "surface/loess"),
           ("leave-one-out", "surface/isotropic_loo"), ("b-spline", "surface/bspline_ls"),
           ("poisson", "surface/poisson_open")]


def book_data(book, model):
  meta, S = V.reading(book)
  X = V.pca_coords(book, model)
  extent, gx, gy = V.plane(X, GRID)
  panels = {}
  for name, figure in METHODS:
    per = {}
    for e in V.EMOTIONS:
      f = V.saved_field(book, model, figure, e)
      if f:
        per[e] = V.rounded(V.resample(f, gx, gy))
    if len(per) == len(V.EMOTIONS):
      panels[name] = per
  return {
    "title": meta["title"], "n": GRID, "extent": extent,
    "support": V.rounded(V.footprint(X, gx, gy), 2),
    "x": V.rounded(X[:, 0], 4), "y": V.rounded(X[:, 1], 4),
    "scores": {e: V.rounded(S[:, j]) for j, e in enumerate(V.EMOTIONS)},
    "panels": panels,
  }


def main():
  ap = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
  ap.add_argument("--model", default="bge-m3")
  ap.add_argument("--books", nargs="+", default=V.SCORED_BOOKS)
  ap.add_argument("--output", default=os.path.join(HERE, "output", "surface_lab.html"))
  args = ap.parse_args()
  data = {"emotions": V.EMOTIONS,
          "books": {b: book_data(b, args.model) for b in args.books}}
  V.render(os.path.join(HERE, "surface_lab_template.html"), data, args.output)


if __name__ == "__main__":
  main()
