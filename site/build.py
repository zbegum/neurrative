"""Collect what is in books/ and every step's output/ into one JSON the site reads.

The page is deliberately not allowed to contain a number that was typed by hand.
Every figure it links and every metric it prints is discovered here, so a stale
claim shows up as a missing entry rather than as a confident sentence about a
result that no longer exists.

Run from anywhere:

    python site/build.py

Writes site/data.json.
"""

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BOOKS = os.path.join(ROOT, "books")
# Each step keeps its results next to its code, in <folder>/output/. The surface
# methods are one family on the page, however many folders they live in.
STEP_OUTPUTS = ["projection", "arc", "arc_on_surface"]
SURFACE_OUTPUTS = ["surface/points", "surface/kernel", "surface/bspline",
                   "surface/poisson"]
MOOD_OUTPUT = "surface/mood"
SITE = os.path.dirname(os.path.abspath(__file__))

# Anything under a directory named this is a quarantined run: kept on disk for
# the record, but not a current result. The site lists these separately rather
# than hiding them, because "we tried it and it did not work" is a finding too.
DEPRECATED = "_deprecated"

# The one book/model pair the geometry work actually covers. Everything past the
# 2-D projections exists only here, so the site reports this pair and says so,
# rather than implying a breadth of coverage that does not exist.
BOOK, MODEL = "alice_wonderland", "bge-m3"

# The figure families, in the order the story is told: what the embedding looks
# like, what emotion does to it, how the surface is fitted, and what the arc is.
FIGURE_ORDER = [
  ("structure_2d", "2-D structure",
   "Paragraph embeddings projected to a plane (PCA / UMAP / t-SNE), colored by "
   "chapter, reading order or emotion."),
  ("projection_comparison", "Which projection?",
   "PCA, UMAP and t-SNE side by side across neighbourhood scale, scored by "
   "trustworthiness and by chapter separation over a shuffled-label control."),
  ("emotions_3d", "Emotion in 3-D",
   "The same paragraphs with an emotion score on the third axis, or as the color "
   "over a 3-component projection."),
  ("surface", "Emotion surface",
   "Everything that fits z = f(PC1, PC2), one folder per method under surface/: "
   "raw/ is the unsmoothed point cloud (surface/points); the four kernel "
   "smoothers tune hx/hy by k-fold CV, isotropic_loo/ tunes a single h by "
   "leave-one-out and bandwidth_sweep/ walks the ladder (surface/kernel); "
   "bspline_ls*/ is the least-squares B-spline (surface/bspline); poisson_*/ is "
   "screened Poisson reconstruction (surface/poisson)."),
  ("mood", "Mood surface",
   "The six emotions collapsed to one mood value per paragraph on a sadness -> "
   "humor spectrum, one surface fitted to it, and the arc drawn on it as "
   "geodesics (surface/mood)."),
  ("geodesics", "Geodesics",
   "Shortest paths between paragraphs measured along the emotion terrain rather "
   "than across the flat plane, so climbing an emotion costs distance."),
  ("windows", "Windowed series",
   "The book as an ordered sequence of mean-pooled windows: the object every arc "
   "figure draws (arc/)."),
  ("narrative_arc_3d", "Narrative arc (3-D)",
   "The arc with a third axis. Four groups, separated by where the height comes "
   "from: height_from_text (the emotion the window itself carried), "
   "height_from_terrain (what the fitted surface predicts at the arc's "
   "location), mood_axis (six emotions collapsed to one), and curve_smoothing."),
  ("arc_comparison", "Is the arc real?",
   "The validation: the windowed arc against the raw per-paragraph path, and "
   "against a shuffled-reading-order control."),
]

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".svg"}
INTERACTIVE_EXT = {".html"}
DATA_EXT = {".npy", ".npz", ".csv"}


def read_json(path):
  try:
    with open(path) as f:
      return json.load(f)
  except (OSError, ValueError):
    return None


def corpus():
  """Per-book paragraph/chapter counts and the embeddings that exist for it."""
  out = []
  for book in sorted(os.listdir(BOOKS)):
    d = os.path.join(BOOKS, book)
    if not os.path.isdir(d) or book.startswith("."):
      continue

    processed = read_json(os.path.join(d, "processed.json")) or {}
    scores = read_json(os.path.join(d, "paragraph_scores.json"))

    emotions = []
    if scores:
      # Every entry carries the same six keys; the first is representative.
      emotions = sorted(scores[0].get("scores", {}))

    models = []
    emb = os.path.join(d, "embeddings")
    if os.path.isdir(emb):
      for m in sorted(os.listdir(emb)):
        meta = read_json(os.path.join(emb, m, "metadata.json")) or {}
        models.append({
          "name": m,
          "hf_model": meta.get("hf_model"),
          "dim": meta.get("embedding_dimension"),
          "n": meta.get("num_paragraphs"),
        })

    out.append({
      "book": book,
      "paragraphs": len(processed.get("paragraphs", [])),
      "chapters": len(processed.get("chapters", [])),
      "scored": len(scores) if scores else 0,
      "emotions": emotions,
      "models": models,
    })
  return out


def classify(name):
  ext = os.path.splitext(name)[1].lower()
  if ext in IMAGE_EXT:
    return "image"
  if ext in INTERACTIVE_EXT:
    return "interactive"
  if ext in DATA_EXT:
    return "data"
  return None


def artifacts():
  """Every figure on disk, grouped by book / model / figure family.

  Paths are stored relative to the repo root so the page can link them with a
  `../` prefix and work straight off the filesystem, with no server.
  """
  groups = []
  deprecated = []

  for book in [BOOK]:
    for model in [MODEL]:
      # figure family -> [(directory to walk, directory paths are relative to)]
      families = {}
      for step in STEP_OUTPUTS:
        md = os.path.join(ROOT, step, "output", book, model)
        for figure in sorted(os.listdir(md)) if os.path.isdir(md) else []:
          fd = os.path.join(md, figure)
          if os.path.isdir(fd):
            families.setdefault(figure, []).append((fd, fd))
      for step in SURFACE_OUTPUTS:
        md = os.path.join(ROOT, step, "output", book, model)
        for method in sorted(os.listdir(md)) if os.path.isdir(md) else []:
          if os.path.isdir(os.path.join(md, method)):
            families.setdefault("surface", []).append(
              (os.path.join(md, method), md))
      md = os.path.join(ROOT, MOOD_OUTPUT, "output", book, model)
      if os.path.isdir(md):
        families["mood"] = [(md, md)]

      for figure in sorted(families):
        files, notes, metrics, params = [], [], {}, []
        for fd, base in families[figure]:
          for dirpath, dirnames, filenames in os.walk(fd):
            dirnames.sort()
            for fn in sorted(filenames):
              full = os.path.join(dirpath, fn)
              rel = os.path.relpath(full, ROOT)
              if fn == "params.json":
                params.append(rel)
                continue
              if fn.lower().endswith(".md"):
                with open(full) as f:
                  notes.append({"path": rel, "name": fn, "text": f.read()})
                continue
              if fn.endswith(".json"):
                payload = read_json(full)
                if payload is not None:
                  # Keyed by path within the family, not by basename: the four
                  # per-smoother directories each hold a `metrics.json` and a
                  # basename key would leave only whichever was walked last.
                  metrics[os.path.relpath(full, base)] = payload
                continue
              kind = classify(fn)
              if kind:
                # The variant subdirectory, when there is one, is the parameter
                # setting the run used -- worth showing beside the file name.
                sub = os.path.relpath(dirpath, base)
                files.append({
                  "path": rel,
                  "name": fn,
                  "kind": kind,
                  "variant": None if sub == "." else sub,
                })

        entry = {
          "book": book,
          "model": model,
          "figure": figure,
          "files": files,
          "notes": notes,
          "metrics": metrics,
          "params": params,
          "n_images": sum(1 for f in files if f["kind"] == "image"),
          "n_interactive": sum(1 for f in files if f["kind"] == "interactive"),
        }
        (deprecated if figure == DEPRECATED else groups).append(entry)

  return groups, deprecated


def smoothing_summary(groups):
  """Held-out RMSE per smoother per emotion, against the flat-mean baseline.

  The four per-method metrics.json files each report the same baseline, so the
  improvement over it is comparable across methods -- which is the actual claim
  the smoothing work makes ("bandwidth matters, the smoother barely does").
  """
  rows = {}
  for g in groups:
    if g["figure"] != "surface":
      continue
    for fn, payload in g["metrics"].items():
      if not isinstance(payload, list):
        continue
      for r in payload:
        method, emotion = r.get("method"), r.get("emotion")
        if not method or not emotion or "val_rmse" not in r:
          continue
        base = r.get("baseline_rmse")
        if base is None:
          # bandwidth_sweep/*/metrics.json is a ladder: many rows per method,
          # no baseline. The per-method files carry the tuned result.
          continue
        rows.setdefault(emotion, {})[method] = {
          "val_rmse": r["val_rmse"],
          "baseline_rmse": base,
          "params": r.get("params"),
          "improvement": (None if not base else
                          100.0 * (base - r["val_rmse"]) / base),
        }
  return rows


def main():
  groups, deprecated = artifacts()
  data = {
    "corpus": corpus(),
    "figure_order": [{"key": k, "title": t, "blurb": b} for k, t, b in FIGURE_ORDER],
    "groups": groups,
    "deprecated": deprecated,
    "smoothing": smoothing_summary(groups),
    "totals": {
      "images": sum(g["n_images"] for g in groups),
      "interactive": sum(g["n_interactive"] for g in groups),
      "quarantined": sum(len(g["files"]) for g in deprecated),
    },
  }

  # Written as a .js assignment rather than .json so index.html works when it is
  # opened straight off the filesystem: a file:// page is not allowed to fetch()
  # a sibling file, but it is allowed to <script src> one.
  path = os.path.join(SITE, "data.js")
  with open(path, "w") as f:
    f.write("window.DATA = ")
    json.dump(data, f, indent=1)
    f.write(";\n")

  print(f"wrote {path}")
  print(f"  {len(data['corpus'])} books, {len(groups)} figure groups, "
        f"{data['totals']['images']} images, "
        f"{data['totals']['interactive']} interactive, "
        f"{data['totals']['quarantined']} quarantined")


if __name__ == "__main__":
  main()
