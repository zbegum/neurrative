"""
The story map: the narrative arc travelling across an emotion landscape, with the
text it is passing through beside it and the emotion over reading order below.

The landscape is one of the saved surfaces (surface/kernel Gaussian,
surface/bspline, surface/poisson), seen flat as contour bands or in 3-D with the
arc riding on it; the mood surface has its own page (surface/mood/mood_viewer.py).
The arc is the canonical route: the 40-paragraph windows' mean points, lifted
onto the shown surface and joined by exact geodesics on it. Drag the timeline, click the
map, arrows step, space plays.

Needs, for each book: projection/embedding.py, and the surface scripts whose
surfaces should appear (a method with no saved surface is left out).

Example:

python arc_on_surface/story_map.py
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
from windows import DEFAULT_SIZE, DEFAULT_STRIDE

HERE = os.path.dirname(os.path.abspath(__file__))
# method key -> (figure family of its saved surface, the variant its default run writes)
METHODS = {"kernel": ("surface/gaussian_nw", "default"),
           "bspline": ("surface/bspline_ls", "d3_p0_lam1e-06"),
           "poisson": ("surface/poisson_open", "d8_pw4_r0.5_b0.08")}


def book_data(book, model):
  data, S = V.reading(book)
  X = V.pca_coords(book, model)
  extent, gx, gy = V.plane(X)
  support = V.footprint(X, gx, gy)
  fields, routes = {}, {}
  for key, (figure, variant) in METHODS.items():
    per, rts = {}, {}
    for e in V.EMOTIONS:
      f = V.saved_field(book, model, figure, e, variant)
      if f:
        values = V.resample(f, gx, gy)
        per[e] = V.rounded(values)
        # the canonical route on this surface: windows lifted, joined by geodesics
        rts[e] = V.route(X, values, gx, gy, DEFAULT_SIZE, DEFAULT_STRIDE, support)
    if per:
      fields[key], routes[key] = per, rts
  data.update({
    "x": V.rounded(X[:, 0], 4), "y": V.rounded(X[:, 1], 4),
    "timeline": V.timelines(S),
    "n": len(gx), "extent": extent, "support": V.rounded(support, 2),
    "fields": fields,
    "routes": routes,
  })
  return data


def main():
  ap = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
  ap.add_argument("--model", default="bge-m3")
  ap.add_argument("--books", nargs="+", default=V.VIEWER_BOOKS)
  ap.add_argument("--output", default=os.path.join(HERE, "output", "story_map.html"))
  args = ap.parse_args()
  data = {"emotions": V.EMOTIONS,
          "books": {b: book_data(b, args.model) for b in args.books}}
  V.render(os.path.join(HERE, "story_map_template.html"), data, args.output)


if __name__ == "__main__":
  main()
