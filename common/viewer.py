"""What the viewer pages share: loading a book for reading, one common grid over
the PCA plane, resampling the saved surfaces onto it, and writing a page.

The pages (arc_on_surface/story_map.py, surface/surface_lab.py,
arc/curve/arc_viewer.py) read results the step scripts already saved, so they
show exactly what those scripts computed. Each page is one self-contained HTML
file: common/web/viewer.css and viewer.js are inlined at build time.
"""

import glob
import json
import os

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import distance_transform_edt, gaussian_filter1d

import paths
from geometry.smoothers import gaussian_nw

WEB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
EMOTIONS = ["wonder", "curiosity", "humor", "confusion", "danger", "sadness"]
TITLES = {"alice_wonderland": "Alice's Adventures in Wonderland",
          "pride_and_prejudice": "Pride and Prejudice", "hamlet": "Hamlet"}
SCORED_BOOKS = ["alice_wonderland", "pride_and_prejudice"]
VIEWER_BOOKS = ["alice_wonderland"]   # the books the interactive pages show by default
GRID = 110           # common grid per axis
TIMELINE_SIGMA = 8   # paragraphs; smooths the timeline curve only


def rounded(a, d=3):
  return [None if not np.isfinite(v) else round(float(v), d) for v in np.ravel(a)]


def pca_coords(book, model):
  path = os.path.join(paths.out_dir(book, model, paths.STRUCTURE_2D, create=False),
                      "pca", "pca.npy")
  if not os.path.exists(path):
    raise SystemExit(f"no {path}: run projection/embedding.py --book {book} --model {model}")
  return np.load(path)[:, :2].astype(float)


def reading(book):
  """Text, chapter per paragraph, chapter names, and the (n, 6) score matrix."""
  processed = json.load(open(os.path.join(paths.ROOT, "books", book, "processed.json")))
  paras = processed["paragraphs"]
  chapters = {}
  for c in processed["chapters"]:
    t = (c.get("title") or "").strip()
    chapters[str(c["chapter_id"])] = (t if t and not t.startswith("[")
                                      else f"Chapter {c['chapter_number']}")
  scores_path = os.path.join(paths.ROOT, "books", book, "paragraph_scores.json")
  S = None
  if os.path.exists(scores_path):
    by_id = {s["paragraph_id"]: s["scores"] for s in json.load(open(scores_path))}
    S = np.array([[by_id[p["id"]][e] for e in EMOTIONS] for p in paras])
  return {
    "title": TITLES.get(book, book),
    "text": [p["text"] for p in paras],
    "chapter": [p["chapter_id"] for p in paras],
    "chapters": chapters,
  }, S


def timelines(S):
  return {e: rounded(gaussian_filter1d(S[:, j], TIMELINE_SIGMA, mode="nearest"))
          for j, e in enumerate(EMOTIONS)}


def plane(X, n=GRID):
  """The common grid: its extent (PCA units, padded) and x / y axes."""
  lo, hi = X.min(0), X.max(0)
  pad = (hi - lo) * 0.04
  gx = np.linspace(lo[0] - pad[0], hi[0] + pad[0], n)
  gy = np.linspace(lo[1] - pad[1], hi[1] + pad[1], n)
  return [float(gx[0]), float(gx[-1]), float(gy[0]), float(gy[-1])], gx, gy


def footprint(X, gx, gy, h=0.4):
  """Where the paragraphs are: kernel density on the grid divided by the 5th
  percentile of the density at the paragraphs themselves (>= 1 means inside)."""
  mu, sd = X.mean(0), X.std(0)
  Z = (X - mu) / sd
  Q = np.array([((a - mu[0]) / sd[0], (b - mu[1]) / sd[1]) for b in gy for a in gx])
  ones = np.ones(len(X))
  _, sup = gaussian_nw(Z, ones, Q, h, h)
  _, at = gaussian_nw(Z, ones, Z, h, h)
  return np.minimum(sup / np.percentile(at, 5), 9.99)


def resample(npz, gx, gy):
  """A saved field_<emotion>_pca.npz on the common grid.

  Cells outside the field's own mask take the nearest valid value, so the page's
  footprint outline (the same for every method) is what decides the edge."""
  d = np.load(npz, allow_pickle=True)
  Z = np.where(d["mask"].astype(bool), d["Z"], np.nan)
  f = RegularGridInterpolator((d["gy"][:, 0], d["gx"][0, :]), Z,
                              bounds_error=False, fill_value=np.nan)
  Q = np.array([(b, a) for b in gy for a in gx])
  out = f(Q).reshape(len(gy), len(gx))
  bad = ~np.isfinite(out)
  if bad.any() and (~bad).any():
    _, (iy, ix) = distance_transform_edt(bad, return_indices=True)
    out = out[iy, ix]
  return out.ravel()


def saved_field(book, model, figure, emotion):
  """The newest saved field for `figure` (e.g. surface/gaussian_nw), or None."""
  base = paths.out_dir(book, model, figure, create=False)
  hits = sorted(glob.glob(os.path.join(base, "**", f"field_{emotion}_pca.npz"),
                          recursive=True), key=os.path.getmtime)
  return hits[-1] if hits else None


def mood_field(X, S, gx, gy, h=0.2):
  """The mood surface (surface/mood, default blend) on the common grid."""
  from surface.mood import mood as mood_mod
  from surface.mood import surface as mood_surface
  m, order, positions, _ = mood_mod.mood(S, EMOTIONS)
  _, _, _, height_at = mood_surface.fit(X, m, h)
  Q = np.array([(a, b) for b in gy for a in gx])
  return height_at(Q), m, order, list(map(float, positions))


def render(template, data, out):
  """Write `template` with the shared CSS / JS and `data` inlined."""
  html = open(template).read()
  html = html.replace("__HEAD_LIBS__", HEAD_LIBS)
  html = html.replace("__CSS__", open(os.path.join(WEB, "viewer.css")).read())
  html = html.replace("__VIEWER_JS__", open(os.path.join(WEB, "viewer.js")).read())
  html = html.replace("__DATA__", json.dumps(data, separators=(",", ":")))
  os.makedirs(os.path.dirname(out), exist_ok=True)
  with open(out, "w") as f:
    f.write(html)
  print(f"wrote {out} ({os.path.getsize(out) / 1e6:.1f} MB)")


# The page head every viewer uses: d3 as a global, three.js through an import map.
HEAD_LIBS = """<script src="https://cdn.jsdelivr.net/npm/d3@7/dist/d3.min.js"></script>
<script type="importmap">{"imports": {
  "three": "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
  "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"}}</script>"""
