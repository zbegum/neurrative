"""
Trace a book's narrative arc in 3-D: the windowed series projected to three
components with PCA, UMAP and t-SNE, and joined in reading order.

Same series and same drawing as arc_2d.py -- mean-pooled sliding windows, a path
in reading order, O at the opening, X at the ending, arrows for direction -- but
the projection keeps a third component instead of flattening it away. In 2-D, two
windows can land on top of each other only because the plane discarded the
direction that separated them; in 3-D more of that separation survives, so a
loop that looked like a crossing can turn out to pass above itself.

How to read the three:

  PCA    the third axis is the third direction of variance, and the shape is real
         geometry within the variance captured (printed per axis).
  UMAP   neighbourhoods are signal; distances, angles and overall shape are not.
  t-SNE  same as UMAP, more so.

Each projection x coloring writes a static PNG and a rotatable HTML (hover a
window to see its paragraph range, chapter, and the opening of its middle
paragraph). A side-by-side grid of all projections is written too.

Outputs, under <output-dir>/<book>/<model>/:

  windows/w40_s20/series.npz                  the pooled series itself
  arc_3d/<pca|umap|tsne>/arc3d_*.png|.html    one per projection x coloring
  arc_3d/<pca|umap|tsne>/<proj>3d_*.npy       the projected coordinates
  arc_3d/<pca|umap|tsne>/fit3d_*.npz          the fitted B-spline, with --fit
  arc_3d/grid3d_*.png|.html                   all projections side by side

Examples:

python curve/arc_3d.py
python curve/arc_3d.py --methods pca --color progression dominant --fit --show-control
python curve/arc_3d.py --size 15 --stride 3 --elev 30 --azim 45
"""

import argparse
import os
import sys

import numpy as np

# arc/ on the path, for the narrative_arc package one level up.
sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from narrative_arc import cli, paths
from narrative_arc import windows as W
from narrative_arc.colors import resolve_colors
from narrative_arc.curves import fit_tag
from narrative_arc.data import load_book
from narrative_arc.plot_3d import (arc_grid_3d, arc_plot_3d, interactive_grid_3d,
                                   interactive_plot_3d, window_hover)
from narrative_arc.projections import project


def run(book, model, args):
  print(f"\n=== {book} / {model} ===")

  series = W.build(args.data_dir, book, model, args.size, args.stride, args.l2)
  print(f"  {series.stops[-1]} paragraphs -> {len(series)} windows "
        f"(size {args.size}, stride {args.stride})")
  saved = W.save(series, args.output_dir, __file__, args)
  print(f"  series -> {saved}")

  specs = resolve_colors(args.color, series, args.min_score)
  print(f"  coloring by: {', '.join(s.name for s in specs)}")

  projections = project(series.pooled, args.methods, n_components=3,
                        neighbors=args.neighbors, min_dist=args.min_dist,
                        perplexity=args.perplexity, seed=args.seed)

  hover = None
  if not args.no_html:
    paragraphs, chapters = load_book(args.data_dir, book)
    hover = window_hover(series, paragraphs, chapters)

  output_dir = paths.out_dir(args.output_dir, book, model, paths.ARC_3D)
  fit = cli.fit_opts(args)
  win_tag = paths.window_tag(args.size, args.stride, args.l2)
  ftag = fit_tag(fit)
  view = dict(elev=args.elev, azim=args.azim)
  heading = (f"{book} / {model} -- window {args.size}, stride {args.stride}")

  fits = []
  for proj in projections:
    method_dir = os.path.join(output_dir, proj.tag)
    os.makedirs(method_dir, exist_ok=True)
    np.save(os.path.join(method_dir, f"{proj.tag}3d_{win_tag}{proj.suffix}.npy"),
            proj.coords)
    proj_fit = cli.fit_projection(
      proj, fit,
      os.path.join(method_dir, f"fit3d_{proj.tag}_{win_tag}{proj.suffix}{ftag}.npz"))
    fits.append(proj_fit)
    for spec in specs:
      title = f"Narrative arc 3-D -- {proj.name} (colored by {spec.name})\n{heading}"
      stem = os.path.join(
        method_dir, f"arc3d_{proj.tag}_{spec.name}_{win_tag}{proj.suffix}{ftag}")
      arc_plot_3d(proj, spec, title, stem + ".png", fit=proj_fit,
                  show_control=args.show_control, **view)
      if hover is not None:
        interactive_plot_3d(proj, spec, hover, title, stem + ".html",
                            fit=proj_fit, show_control=args.show_control)

  if len(projections) > 1 and not args.no_grid:
    for spec in specs:
      title = f"Narrative arc 3-D (colored by {spec.name}) -- {heading}"
      stem = os.path.join(output_dir, f"grid3d_{spec.name}_{win_tag}"
                                      f"{cli.grid_suffix(projections)}{ftag}")
      arc_grid_3d(projections, spec, title, stem + ".png", fits=fits,
                  show_control=args.show_control, **view)
      if hover is not None:
        interactive_grid_3d(projections, spec, hover, title, stem + ".html",
                            fits=fits, show_control=args.show_control)

  paths.stamp(output_dir, __file__, args, book=book, model=model,
              n_windows=len(series), colorings=[s.name for s in specs],
              projections={p.tag: p.info for p in projections})
  print(f"  -> {output_dir}")


def main():
  parser = argparse.ArgumentParser(
    description="Narrative arc in 3-D: windowed embeddings projected to three "
                "components with PCA, UMAP and t-SNE and joined in reading order.")
  cli.add_arc_args(parser)
  parser.add_argument("--elev", default=22.0, type=float,
                      help="Static PNG camera elevation, in degrees.")
  parser.add_argument("--azim", default=-60.0, type=float,
                      help="Static PNG camera azimuth, in degrees.")
  parser.add_argument("--no-html", action="store_true",
                      help="Skip the interactive plotly HTML.")
  args = parser.parse_args()

  for book, model in cli.targets(args, parser):
    run(book, model, args)
  print("\nDone.")


if __name__ == "__main__":
  main()
