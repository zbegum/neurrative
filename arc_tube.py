"""
The narrative arc as a tube: a polygon cross-section at every window, swept
smoothly along the fitted 3-D curve.

The arc is where the story goes; the section is what the story is like there.
Joining the sections in reading order gives a surface whose shape changes in
time -- see narrative_arc/tube.py for the construction (rotation-minimizing
frames, PCHIP sweep, closed mesh).

Sections (--section):

  emotions  one vertex per emotion (a hexagon for six). A vertex's distance from
            the curve grows with the window's pooled score, so the tube bulges on
            the side of whichever emotion is strong. Any projection.
  spread    the window's paragraphs, placed with the same PCA, projected onto the
            plane perpendicular to the curve: their covariance ellipse, sampled as
            a --sides polygon. Standard error by default (how precisely the window
            is located); --spread-scale sd for the paragraphs' own scatter. PCA only.

Outputs, under <output-dir>/<book>/<model>/arc_tube/<pca|umap|tsne>/:

  tube_*.png    static render
  tube_*.html   rotatable, hover a window for its text
  tube_*.npz    vertices, faces, per-sample radii, window polygons

Examples:

python arc_tube.py
python arc_tube.py --section emotions --normalize --round 36
python arc_tube.py --section spread --sides 16 --color progression
python arc_tube.py --data-dir ../neurrative/books --book hamlet --section spread
"""

import argparse
import os

import numpy as np

from narrative_arc import cli, paths
from narrative_arc import tube as TB
from narrative_arc import windows as W
from narrative_arc.curves import FitOpts, fit_curve, fit_tag
from narrative_arc.data import load_book, load_embeddings
from narrative_arc.plot_3d import window_hover
from narrative_arc.plot_tube import tube_html, tube_plot
from narrative_arc.projections import METHODS, project

ARC_TUBE = "arc_tube"
N_SAMPLES = 400


def tube_tag(args, thickness):
  tag = f"_{args.section}-t{thickness:g}-sm{args.smooth:g}"
  if args.section == "emotions":
    tag += f"-c{args.core:g}" + ("-norm" if args.normalize else "")
  else:
    tag += f"-k{args.sides}-{args.spread_scale}"
  if args.round:
    tag += f"-r{args.round}"
  return tag


def run(book, model, args):
  print(f"\n=== {book} / {model} ===")
  series = W.build(args.data_dir, book, model, args.size, args.stride, args.l2)
  n = len(series)
  print(f"  {n} windows (size {args.size}, stride {args.stride})")

  section = args.section or ("emotions" if series.scores is not None else "spread")
  args.section = section
  thickness = args.thickness if args.thickness is not None else (
    0.35 if section == "emotions" else 1.0)
  color = args.color or ("emotion" if section == "emotions" else "progression")

  paragraphs, chapters = load_book(args.data_dir, book)
  hover = window_hover(series, paragraphs, chapters)
  embeddings = load_embeddings(args.data_dir, book, model) if section == "spread" else None
  win_tag = paths.window_tag(args.size, args.stride, args.l2)
  output_dir = paths.out_dir(args.output_dir, book, model, ARC_TUBE)

  for method in args.methods:
    if section == "spread" and method != "pca":
      print(f"  skipping {method}: spread sections need PCA")
      continue
    proj = project(series.pooled, [method], n_components=3,
                   neighbors=min(args.neighbors, n - 1), min_dist=args.min_dist,
                   perplexity=args.perplexity, seed=args.seed)[0]

    opts = FitOpts(args.n_control or int(np.clip(n // 4, 6, 24)), args.degree,
                   args.lambda_, "dn", 300)
    fit = fit_curve(proj.coords, opts, n_eval=N_SAMPLES)
    print(f"  {proj.name} fit: {opts.n_control} control points, lambda "
          f"{opts.lambda_:g}, {'converged' if fit.converged else 'NOT converged'}")

    frames = TB.rotation_minimizing_frames(fit.curve)
    _, window_index = TB.window_positions(fit, N_SAMPLES, opts.degree)
    rms = np.sqrt(((proj.coords - proj.coords.mean(axis=0)) ** 2).sum(axis=1).mean())

    if section == "emotions":
      radii, labels = TB.emotion_sections(series, rms, thickness, args.core,
                                          args.normalize)
    else:
      radii, labels = TB.spread_sections(proj, embeddings, series, frames[1],
                                         frames[2], window_index, args.sides,
                                         args.spread_scale, thickness)
    tube = TB.build_tube(fit, radii, opts.degree, labels, args.smooth, args.round,
                         frames, args.fold_limit)
    print(f"  tube: {len(tube.vertices)} vertices, {len(tube.faces)} faces; "
          f"radius / layout RMS radius {radii.min() / rms:.3f} - "
          f"{radii.max() / rms:.3f}; whole tube scaled x{tube.scale:.2f} and "
          f"{tube.clamped:.0%} of the curve narrowed further, to avoid folding")

    method_dir = os.path.join(output_dir, proj.tag)
    os.makedirs(method_dir, exist_ok=True)
    stem = os.path.join(method_dir, f"tube_{proj.tag}_{win_tag}{proj.suffix}"
                                    f"{tube_tag(args, thickness)}{fit_tag(opts)}")
    np.savez(stem + ".npz", vertices=tube.vertices, faces=tube.faces,
             radii=tube.radii, window_rings=tube.window_rings,
             window_radii=radii, window_index=tube.window_index,
             angles=tube.angles, labels=np.array(labels or []),
             curve=fit.curve)

    what = ("emotion sections" if section == "emotions"
            else f"paragraph spread ({args.spread_scale})")
    title = (f"Narrative tube -- {proj.name}, {what}\n"
             f"{book} / {model} -- window {args.size}, stride {args.stride}")
    tube_plot(tube, proj, fit.curve, title, stem + ".png", color,
              not args.no_rings, args.elev, args.azim)
    if not args.no_html:
      tube_html(tube, proj, fit.curve, hover, title, stem + ".html", color,
                not args.no_rings)

  paths.stamp(output_dir, __file__, args, book=book, model=model, n_windows=n)
  print(f"  -> {output_dir}")


def main():
  parser = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
  cli.add_target_args(parser)
  cli.add_window_args(parser)
  parser.add_argument("--methods", nargs="+", default=["pca"], choices=METHODS)
  parser.add_argument("--section", choices=["emotions", "spread"],
                      help="Default: emotions when the book has scores, else spread.")
  parser.add_argument("--thickness", type=float,
                      help="emotions: largest radius as a fraction of the layout's "
                           "RMS radius (default 0.35). spread: multiplier on the "
                           "ellipse (default 1).")
  parser.add_argument("--core", default=0.15, type=float,
                      help="emotions: radius left at score 0, as a fraction of the "
                           "full radius, so the tube never pinches to a line.")
  parser.add_argument("--normalize", action="store_true",
                      help="emotions: rescale each emotion to its own range over "
                           "the book -- show change, not absolute strength.")
  parser.add_argument("--sides", default=12, type=int,
                      help="spread: polygon vertices around the ellipse.")
  parser.add_argument("--spread-scale", default="se", choices=["se", "sd"],
                      help="spread: standard error of the window mean, or the "
                           "paragraphs' standard deviation.")
  parser.add_argument("--smooth", default=1.0, type=float,
                      help="Gaussian smoothing of sections across windows, in "
                           "windows (0 = none).")
  parser.add_argument("--fold-limit", default=0.9, type=float,
                      help="Largest section radius as a fraction of the local "
                           "radius of curvature. The whole tube is scaled so 95%% "
                           "of the curve respects it, then the sharpest bends are "
                           "narrowed locally (0 = off, allow folds).")
  parser.add_argument("--round", default=0, type=int,
                      help="Resample each section to this many vertices with a "
                           "periodic spline (0 = keep the polygon).")
  parser.add_argument("--color", choices=["emotion", "progression"],
                      help="Default: emotion for emotion sections, else progression.")
  parser.add_argument("--n-control", type=int,
                      help="Curve control points (default: windows / 4, 6-24 -- a "
                           "smooth centerline, so the tube can be thick without "
                           "folding at tight bends).")
  parser.add_argument("--degree", default=3, type=int)
  parser.add_argument("--lambda", dest="lambda_", default=0.1, type=float)
  parser.add_argument("--seed", default=0, type=int)
  parser.add_argument("--neighbors", default=15, type=int)
  parser.add_argument("--min-dist", default=0.1, type=float)
  parser.add_argument("--perplexity", default=30.0, type=float)
  parser.add_argument("--no-rings", action="store_true",
                      help="Hide the polygon outline at each window.")
  parser.add_argument("--no-html", action="store_true")
  parser.add_argument("--elev", default=22.0, type=float)
  parser.add_argument("--azim", default=-60.0, type=float)
  args = parser.parse_args()

  for book, model in cli.targets(args, parser):
    run(book, model, args)
  print("\nDone.")


if __name__ == "__main__":
  main()
