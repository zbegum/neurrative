"""
The story map: one page where the narrative arc travels across an emotion
landscape while the text it is passing through is shown beside it.

Three linked views: the PCA plane with one emotion's landscape (Gaussian
Nadaraya-Watson, the bandwidth the kernel step tuned) and the arc of
40-paragraph windows over it; the paragraph text; and the emotion over reading
order. Dragging the timeline moves the marker along the arc; clicking the map
jumps to the nearest paragraph.

Books without emotion scores (Hamlet) are skipped. Reads the PCA coordinates
from projection/ and the tuned bandwidths from surface/kernel/ when they exist.

Example:

python arc_on_surface/story_map.py
"""

import argparse
import json
import os
import sys

import numpy as np
from scipy.ndimage import gaussian_filter1d

# The repository root and common/ on the path, wherever this script lives.
_ROOT = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_ROOT, "common")):
  _ROOT = os.path.dirname(_ROOT)
sys.path[1:1] = [_ROOT, os.path.join(_ROOT, "common")]

import paths
from geometry.smoothers import gaussian_nw
from windows import DEFAULT_SIZE, DEFAULT_STRIDE, window_bounds

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(HERE, "story_map_template.html")
TITLES = {"alice_wonderland": "Alice's Adventures in Wonderland",
          "pride_and_prejudice": "Pride and Prejudice"}
EMOTIONS = ["wonder", "curiosity", "humor", "confusion", "danger", "sadness"]
GRID = 120           # landscape resolution per axis
TIMELINE_SIGMA = 8   # paragraphs; smooths the timeline curve only


def pca_coords(book, model):
  path = os.path.join(paths.out_dir(book, model, paths.STRUCTURE_2D, create=False),
                      "pca", "pca.npy")
  if os.path.exists(path):
    return np.load(path)[:, :2].astype(float)
  from sklearn.decomposition import PCA
  emb = np.load(os.path.join(paths.ROOT, "books", book, "embeddings", model,
                             "embeddings.npy"))
  return PCA(n_components=2).fit_transform(emb)


def tuned_bandwidths(book, model):
  """{emotion: (hx, hy)} from surface/kernel's Gaussian run, if it has been run."""
  out = {}
  f = os.path.join(paths.out_dir(book, model, "surface/gaussian_nw", create=False),
                   "default", "metrics.json")
  if os.path.exists(f):
    for r in json.load(open(f)):
      if r.get("baseline_rmse") is not None:
        out[r["emotion"]] = (r["params"]["hx"], r["params"]["hy"])
  return out


def landscape(X, y, hx, hy):
  """Gaussian NW over the PCA box, in standardized coordinates.

  Returns the field, the kernel support divided by the 5th percentile of the
  support at the paragraphs (the page clips the map where this is below 1, so
  the edge follows the paragraphs smoothly), and the extent in PCA units."""
  mu, sd = X.mean(0), X.std(0)
  Z = (X - mu) / sd
  lo, hi = Z.min(0), Z.max(0)
  pad = (hi - lo) * 0.04
  gx = np.linspace(lo[0] - pad[0], hi[0] + pad[0], GRID)
  gy = np.linspace(lo[1] - pad[1], hi[1] + pad[1], GRID)
  Q = np.array([(a, b) for b in gy for a in gx])
  val, sup = gaussian_nw(Z, y, Q, hx, hy)
  _, sup_pts = gaussian_nw(Z, y, Z, hx, hy)
  support = sup / np.percentile(sup_pts, 5)
  extent = [float(gx[0] * sd[0] + mu[0]), float(gx[-1] * sd[0] + mu[0]),
            float(gy[0] * sd[1] + mu[1]), float(gy[-1] * sd[1] + mu[1])]
  return val.reshape(GRID, GRID), support.reshape(GRID, GRID), extent


def r(a, d=4):
  return [None if not np.isfinite(v) else round(float(v), d) for v in np.ravel(a)]


def book_data(book, model):
  processed = json.load(open(os.path.join(paths.ROOT, "books", book, "processed.json")))
  scores = {s["paragraph_id"]: s["scores"]
            for s in json.load(open(os.path.join(paths.ROOT, "books", book,
                                                 "paragraph_scores.json")))}
  paras = processed["paragraphs"]
  S = np.array([[scores[p["id"]][e] for e in EMOTIONS] for p in paras])
  X = pca_coords(book, model)
  bw = tuned_bandwidths(book, model)

  chapters = {}
  for c in processed["chapters"]:
    t = (c.get("title") or "").strip()
    chapters[c["chapter_id"]] = (t if t and not t.startswith("[")
                                 else f"Chapter {c['chapter_number']}")

  fields, supports, extent = {}, {}, None
  for j, e in enumerate(EMOTIONS):
    hx, hy = bw.get(e, (0.4, 0.4))
    Z, sup, extent = landscape(X, S[:, j], hx, hy)
    fields[e], supports[e] = r(Z, 3), r(np.minimum(sup, 9.99), 2)

  bounds = window_bounds(len(paras), DEFAULT_SIZE, DEFAULT_STRIDE)
  arc = [[float(X[a:b, 0].mean()), float(X[a:b, 1].mean()), (a + b - 1) / 2.0]
         for a, b in bounds]

  return {
    "title": TITLES.get(book, book),
    "x": r(X[:, 0]), "y": r(X[:, 1]),
    "text": [p["text"] for p in paras],
    "chapter": [p["chapter_id"] for p in paras],
    "chapters": {str(k): v for k, v in chapters.items()},
    "timeline": {e: r(gaussian_filter1d(S[:, j], TIMELINE_SIGMA, mode="nearest"), 3)
                 for j, e in enumerate(EMOTIONS)},
    "grid": GRID, "extent": extent, "fields": fields, "support": supports,
    "arc": [[round(a, 4), round(b, 4), c] for a, b, c in arc],
  }


def main():
  ap = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
  ap.add_argument("--model", default="bge-m3")
  ap.add_argument("--books", nargs="+", default=list(TITLES))
  ap.add_argument("--output", default=None,
                  help="Default: arc_on_surface/output/story_map.html")
  args = ap.parse_args()

  data = {"emotions": EMOTIONS,
          "books": {b: book_data(b, args.model) for b in args.books}}
  html = open(TEMPLATE).read().replace("__DATA__", json.dumps(data, separators=(",", ":")))
  out = args.output or os.path.join(HERE, "output", "story_map.html")
  os.makedirs(os.path.dirname(out), exist_ok=True)
  with open(out, "w") as f:
    f.write(html)
  print(f"wrote {out} ({os.path.getsize(out) / 1e6:.1f} MB)")


if __name__ == "__main__":
  main()
