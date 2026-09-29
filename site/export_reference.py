"""Extract the pipeline reference from the source: flow, parameters, algorithms.

Everything mechanical is read out of the code with `ast` rather than transcribed
-- CLI flags and their defaults, which `paths.*` directory a script writes to,
which repo modules it imports, and the parameter search grids. A default that
changes in the source changes on the page at the next build, instead of quietly
disagreeing with it.

The one hand-authored part is PROVENANCE below: where an algorithm came from is
knowledge about the code, not something in it.

    python site/export_reference.py    ->    site/reference.js
"""

import ast
import glob
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SITE = os.path.dirname(os.path.abspath(__file__))
GEO = os.path.join(ROOT, "geometry")
COMMON = os.path.join(ROOT, "common")
# Every folder that holds runnable scripts, in pipeline order.
SCRIPT_DIRS = ["preprocess", "common", "projection", "surface/points",
               "surface/kernel", "surface/bspline", "surface/poisson",
               "surface/mood", "arc/validation", "arc_on_surface"]

# The book and model everything below is reported for.
BOOK, MODEL = "alice_wonderland", "bge-m3"

# Where each algorithm actually comes from. Verified against the vendored
# LICENSE/PROVENANCE files and the model registry, not from memory.
PROVENANCE = [
  {
    "name": "Exact geodesics (MMP)",
    "what": "Exact shortest paths on a triangle mesh. Propagate a distance "
            "field over the surface, then trace back from target to source.",
    "impl": "geometry/geodesic_cpp/ (vendored C++), driven by "
            "geometry/_geodesic_native.py via ctypes",
    "origin": "Danil Kirsanov's `geodesic` library, 01/2008 — vendored verbatim. "
              "Our only addition is wrapper.cpp, the extern \"C\" ABI.",
    "links": [
      ["Upstream (Google Code archive)", "https://code.google.com/archive/p/geodesic/"],
      ["Mitchell, Mount & Papadimitriou 1987, SIAM J. Comput. 16:647–668",
       "https://doi.org/10.1137/0216045"],
    ],
    "note": "Ships with Dijkstra and subdivision approximations too, but "
            "wrapper.cpp never references them — only the exact solver is used. "
            "Built on first import; needs a C++ compiler, and there is no "
            "fallback if one is missing.",
  },
  {
    "name": "Distance-based curve smoothing",
    "what": "Smooth a curve by minimizing a Dirichlet energy lifted through a "
            "distance field, so closeness to the original is one continuous "
            "parameter tau.",
    "impl": "geometry/curve_smoothing.py — our implementation, specialized to a "
            "flat planar domain (no 4-D lift, no mesh geodesic solver needed). "
            "Solved with L-BFGS-B on an analytic gradient.",
    "origin": "Pawellek, Rössl & Lawonn, \"Distance-Based Smoothing of Curves "
              "on Surface Meshes\", Computer Graphics Forum, 2024. Cited by "
              "DOI; the PDF is not redistributed.",
    "links": [["Computer Graphics Forum 2024",
               "https://doi.org/10.1111/cgf.15015"]],
    "note": "Replaced the retired control-point fit (arc_geodesic.py).",
  },
  {
    "name": "B-spline curve fitting",
    "what": "Uniform B-spline regression through the windowed arc, in 2-D "
            "or 3-D: control points and each window's position on the curve "
            "are optimised jointly.",
    "impl": "arc/vendor/bspline_regression/ (vendored Python), "
            "set up by arc/narrative_arc/curves.py",
    "origin": "github.com/rstebbing/bspline-regression, MIT. Only the files on "
              "the fitting path are vendored; see its PROVENANCE.md.",
    "links": [["rstebbing/bspline-regression",
               "https://github.com/rstebbing/bspline-regression"]],
    "note": "Replaced the earlier JS bridge (mirsaeedi/spline-curve-fitting "
            "via fit_bspline.js) together with the old 2-D arc script.",
  },
  {
    "name": "Nadaraya–Watson kernel regression",
    "what": "Degree-0 local regression: the surface height at a point is a "
            "kernel-weighted average of nearby paragraph scores.",
    "impl": "One implementation: geometry/smoothers.py:nadaraya_watson, "
            "dimension-general, with the bandwidth given per axis or as one "
            "value for all of them. gaussian_nw(hx, hy) is the two-axis "
            "spelling used by the tuning protocol; geometry/scalar_field.py "
            "is the isotropic face, adding a grid and closed-form "
            "leave-one-out bandwidth selection on top of the same estimator.",
    "origin": "Standard method (Nadaraya 1964; Watson 1964). Written here, not "
              "vendored.",
    "links": [],
    "note": "Previously implemented twice. Consolidated onto smoothers.py; the "
            "two entry points now differ only in how the bandwidth is chosen "
            "and in what units it is expressed, which is a real difference. "
            "Verified: tuned parameters and held-out RMSE unchanged for all six "
            "emotions, and the LOO-selected h is identical.",
  },
  {
    "name": "Local polynomial regression / LOESS",
    "what": "Degree-1 or -2 local fits. `local_linear` uses a fixed Gaussian "
            "bandwidth; `loess` uses an adaptive k-NN window, tricube weights "
            "and bisquare robustness iterations.",
    "impl": "geometry/smoothers.py — written here (batched weighted normal "
            "equations via einsum, with a 1e-12 ridge).",
    "origin": "Cleveland 1979 (LOWESS); Cleveland & Devlin 1988.",
    "links": [["Cleveland 1979, JASA 74:829–836",
               "https://doi.org/10.1080/01621459.1979.10481038"]],
    "note": "",
  },
  {
    "name": "PCA / t-SNE / UMAP",
    "what": "Dimensionality reduction from the embedding to a 2-D or 3-D plane.",
    "impl": "scikit-learn (PCA, TSNE) and umap-learn (UMAP), called from "
            "projection/embedding.py and friends.",
    "origin": "Third-party libraries, unmodified.",
    "links": [
      ["umap-learn", "https://github.com/lmcinnes/umap"],
      ["scikit-learn manifold", "https://scikit-learn.org/stable/modules/manifold.html"],
    ],
    "note": "PCA is the only one of the three that supports a height field: it "
            "is linear and metric, so a point has a well-defined position. "
            "UMAP/t-SNE are non-metric.",
  },
  {
    "name": "Text embeddings",
    "what": "One vector per paragraph.",
    "impl": "preprocess/models/ — thin wrappers over HuggingFace checkpoints. "
            "All outputs L2-normalized.",
    "origin": "BAAI/bge-m3 (1024-d, used for everything here); "
              "intfloat/e5-large-v2 (1024-d); Qwen/Qwen3-Embedding-4B (2560-d).",
    "links": [
      ["BAAI/bge-m3", "https://huggingface.co/BAAI/bge-m3"],
      ["intfloat/e5-large-v2", "https://huggingface.co/intfloat/e5-large-v2"],
      ["Qwen/Qwen3-Embedding-4B", "https://huggingface.co/Qwen/Qwen3-Embedding-4B"],
    ],
    "note": "nv-embed-v2 and gte-qwen2 are wired up but commented out of the "
            "registry (they need transformers 4.51.3).",
  },
  {
    "name": "Emotion annotation",
    "what": "Six scores in [0,1] per paragraph — wonder, danger, sadness, "
            "humor, confusion, curiosity — plus a summary, keywords and "
            "characters.",
    "impl": "preprocess/get_annotations.py — greedy decoding "
            "(do_sample=False, max_new_tokens=256), bfloat16, device_map=auto.",
    "origin": "Qwen/Qwen2.5-7B-Instruct (the --model default).",
    "links": [["Qwen2.5-7B-Instruct",
               "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct"]],
    "note": "No model id is recorded in paragraph_scores.json, so the annotator "
            "behind a committed score file cannot be verified from the file.",
  },
]

# The reading order of the pipeline. `script` keys are matched against the
# extracted CLI table; stages with no script are data-preparation steps.
FLOW = [
  ("ingest", "Ingest", ["preprocess/preprocess.py", "preprocess/preprocess_play.py"],
   "raw.txt → processed.json (chapters, paragraphs)"),
  ("embed", "Embed", ["preprocess/get_embeddings.py"],
   "processed.json → embeddings.npy (789 × 1024 for bge-m3)"),
  ("annotate", "Annotate", ["preprocess/get_annotations.py"],
   "processed.json → paragraph_scores.json (6 emotions)"),
  ("project", "Project", ["projection/embedding.py", "projection/emotion_3d.py",
                          "projection/projection_comparison.py",
                          "projection/chain_umap.py"],
   "embeddings.npy → 2-D / 3-D coordinates"),
  ("fit", "Fit the landscape",
   ["surface/points/raw_points.py",
    "surface/kernel/gaussian.py", "surface/kernel/epanechnikov.py",
    "surface/kernel/local_linear.py", "surface/kernel/loess.py",
    "surface/kernel/loo.py",
    "surface/kernel/bandwidth_grid.py", "surface/kernel/smooth_grid.py",
    "surface/bspline/fit_surface.py", "surface/poisson/fit_surface.py",
    "surface/mood/run.py"],
   "raw scores over the PCA plane → z = f(PC1, PC2), one folder per method "
   "under surface/"),
  ("arc", "Trace the arc",
   ["common/windows.py"],
   "sliding window over reading order → the windowed series; drawn, fitted "
   "and swept into a tube by arc/"),
  ("validate", "Validate the arc",
   ["arc/validation/arc_comparison.py",
    "arc/validation/arc_comparison_projections.py"],
   "arc vs raw path, and vs a shuffled-order control"),
  ("arc3d", "The arc in 3-D",
   ["arc_on_surface/narrative_arc_3d.py", "arc_on_surface/arc_on_surface.py",
    "arc_on_surface/arc_emotion_axis.py", "arc_on_surface/arc_smooth.py",
    "arc_on_surface/arc_emotion_smoother_grid.py"],
   "the arc + a third axis → narrative_arc_3d/{height_from_text,"
   "height_from_terrain,mood_axis,curve_smoothing}/"),
  ("geodesic", "Geodesics",
   ["arc_on_surface/geodesic_arrows.py", "arc_on_surface/geodesic_interactive.py"],
   "paths measured along the terrain, not across the plane"),
]


def literal(node):
  """Best-effort constant value of an AST node, or a source-ish string."""
  try:
    return ast.literal_eval(node)
  except (ValueError, SyntaxError):
    try:
      return ast.unparse(node)
    except Exception:
      return None


def parse_script(path):
  """CLI flags, imported repo modules, and the paths.* constant a script uses."""
  src = open(path).read()
  tree = ast.parse(src)

  # First non-blank line of the module docstring. Left empty when there is none,
  # so the page can show the gap rather than invent a description.
  doc = ast.get_docstring(tree) or ""
  purpose = next((l.strip() for l in doc.splitlines() if l.strip()), "")

  args, imports, out_consts, grids = [], set(), set(), {}
  read_consts, inherits = set(), set()

  # A paths.* constant inside a load_*/read_* helper is an input, not an output:
  # smooth_common reads structure_2d/pca/pca.npy there and writes somewhere else
  # entirely, and conflating the two makes every smoother look like it emits
  # projections.
  # The geodesic scripts import the module as `out_paths`, because `paths` is
  # already a local variable there (the polylines). Follow the alias rather than
  # assuming the module is always bound to its own name.
  alias = "paths"
  for node in ast.walk(tree):
    if isinstance(node, ast.Import):
      for a in node.names:
        if a.name == "paths" and a.asname:
          alias = a.asname

  reader_lines = set()
  for fn in ast.walk(tree):
    if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) and \
       (fn.name.startswith("load") or fn.name.startswith("read")):
      for sub in ast.walk(fn):
        if hasattr(sub, "lineno"):
          reader_lines.add(sub.lineno)

  for node in ast.walk(tree):
    # Shared flags are added by a helper, not by this file. The four smoother
    # wrappers are ~40 lines each and would otherwise look argument-less while
    # actually taking the whole common protocol (--val-frac, --folds, --seed...).
    if isinstance(node, ast.Call):
      fn = node.func
      fname = fn.attr if isinstance(fn, ast.Attribute) else \
              fn.id if isinstance(fn, ast.Name) else None
      if fname in ("add_common_args", "run"):
        inherits.add("common/smooth_common.py")

    # --flags
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
       and node.func.attr == "add_argument":
      names = [literal(a) for a in node.args]
      flag = next((n for n in names if isinstance(n, str) and n.startswith("--")), None)
      # Positional arguments have no leading dashes -- preprocess.py's `book` is
      # one, and skipping them would show that script as having no interface.
      positional = flag is None and names and isinstance(names[0], str)
      if positional:
        flag = names[0]
      if not flag:
        continue
      kw = {k.arg: literal(k.value) for k in node.keywords}
      args.append({
        "flag": flag,
        "positional": bool(positional),
        "required": bool(kw.get("required")) or bool(positional),
        "default": kw.get("default"),
        "choices": kw.get("choices"),
        "nargs": kw.get("nargs"),
        "action": kw.get("action"),
        "help": (kw.get("help") or "").strip() or None,
      })

    # repo modules this script builds on
    if isinstance(node, ast.ImportFrom) and node.module:
      imports.add(node.module.split(".")[0])
    elif isinstance(node, ast.Import):
      for a in node.names:
        imports.add(a.name.split(".")[0])

    # which output directory it writes to
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) \
       and node.value.id == alias and node.attr.isupper():
      target = read_consts if node.lineno in reader_lines else out_consts
      target.add(node.attr)

    # parameter search grids (GRID = {...} at module level)
    if isinstance(node, ast.Assign):
      for t in node.targets:
        if isinstance(t, ast.Name) and t.id.endswith("GRID"):
          val = literal(node.value)
          if isinstance(val, dict):
            grids[t.id] = val

  repo = {"data", "geometry", "smooth_common", "paths", "narrative_arc",
          "arc_on_surface", "arc_emotion_axis", "arc_emotion_smoother_grid",
          "geodesic_arrows", "models"}
  return {
    "purpose": purpose,
    "args": args,
    "inherits": sorted(inherits),
    "imports": sorted(imports & repo),
    "writes": sorted(out_consts - read_consts),
    "reads": sorted(read_consts),
    "grids": grids,
    "lines": src.count("\n") + 1,
  }


def resolve_inherited(scripts):
  """Fold a helper's flags into the scripts that call it.

  Done after every file is parsed, because the helper may be parsed second.
  Inherited entries are marked so the page can show where a flag really lives
  rather than implying the 40-line wrapper declares it.
  """
  for path, s in scripts.items():
    own = {a["flag"] for a in s["args"]}
    for src_path in s.get("inherits", ()):
      donor = scripts.get(src_path)
      if not donor or src_path == path:
        continue
      for a in donor["args"]:
        if a["flag"] not in own:
          s["args"].append({**a, "from": src_path})
          own.add(a["flag"])
      # ...and the directories the helper touches on their behalf.
      if not s["writes"]:
        s["writes"] = list(donor["writes"])
      if not s.get("reads"):
        s["reads"] = list(donor.get("reads", []))
  return scripts


def path_constants():
  """The paths.* constants and the directory names they resolve to."""
  tree = ast.parse(open(os.path.join(COMMON, "paths.py")).read())
  out = {}
  for node in tree.body:
    if isinstance(node, ast.Assign):
      for t in node.targets:
        if isinstance(t, ast.Name) and t.id.isupper():
          v = literal(node.value)
          if isinstance(v, str):
            out[t.id] = v
  return out


def chosen_params():
  """What the tuning actually picked for this book/model, per method/emotion."""
  picked = {}
  # <method>/<variant>/metrics.json under each surface folder's output.
  for base in sorted(glob.glob(os.path.join(ROOT, "surface", "*", "output",
                                            BOOK, MODEL))):
    for f in sorted(glob.glob(os.path.join(base, "**", "metrics.json"),
                              recursive=True)):
      method = os.path.relpath(f, base).split(os.sep)[0]
      try:
        rows = json.load(open(f))
      except ValueError:
        continue
      if not isinstance(rows, list):
        continue
      for r in rows:
        if r.get("baseline_rmse") is None:
          continue
        picked.setdefault(method, {})[r["emotion"]] = r.get("params")
  return picked


def main():
  scripts = {}
  for prefix in SCRIPT_DIRS:
    folder = os.path.join(ROOT, prefix)
    for fn in sorted(os.listdir(folder)):
      if not fn.endswith(".py"):
        continue
      full = os.path.join(folder, fn)
      if os.path.getsize(full) == 0:
        continue
      key = f"{prefix}/{fn}"
      try:
        scripts[key] = parse_script(full)
      except SyntaxError:
        continue

  resolve_inherited(scripts)

  # module sizes for geometry/, which has no CLI of its own
  geometry = {}
  for fn in sorted(os.listdir(GEO)):
    if fn.endswith(".py") and os.path.getsize(os.path.join(GEO, fn)):
      info = parse_script(os.path.join(GEO, fn))
      geometry[f"geometry/{fn}"] = {"purpose": info["purpose"], "lines": info["lines"]}

  data = {
    "book": BOOK,
    "model": MODEL,
    "flow": [{"key": k, "title": t, "scripts": s, "io": io} for k, t, s, io in FLOW],
    "scripts": scripts,
    "geometry": geometry,
    "paths": path_constants(),
    "provenance": PROVENANCE,
    "chosen": chosen_params(),
  }

  out = os.path.join(SITE, "reference.js")
  with open(out, "w") as f:
    f.write("window.REF = ")
    json.dump(data, f, indent=1)
    f.write(";\n")

  n_args = sum(len(s["args"]) for s in scripts.values())
  print(f"wrote {out}")
  print(f"  {len(scripts)} scripts, {n_args} CLI flags, "
        f"{len(geometry)} geometry modules, {len(PROVENANCE)} algorithms")


if __name__ == "__main__":
  main()
