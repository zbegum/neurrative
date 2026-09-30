"""
Trace a book's narrative arc in 2-D by sliding a window over its paragraphs.

Row i of embeddings.npy is paragraph i in reading order, so a sliding window is
just a mean-pool over consecutive rows: window k averages paragraphs
[k*stride, k*stride + size). Pooling smooths out paragraph-to-paragraph jitter
and, crucially, keeps the temporal order -- window k+1 always follows window k --
so when the pooled points are projected to 2-D and connected in order, the line
is the story's path through semantic space. A comedy tends to loop back near
where it started; a tragedy drifts away and never returns.

The window is drawn three ways -- PCA, UMAP, t-SNE -- since no single projection
is trustworthy on its own; agreement across them is the signal. Each path is
colored by progression (start -> end) by default, with the opening marked O and
the ending X, and short arrows along the line to show direction.

Outputs, under <output-dir>/<book>/<model>/:

  windows/w40_s20/series.npz               the pooled series itself
  arc_2d/<pca|umap|tsne>/arc_*.png         one figure per projection x coloring
  arc_2d/<pca|umap|tsne>/<proj>_*.npy      the projected coordinates
  arc_2d/<pca|umap|tsne>/fit_*.npz         the fitted B-spline, with --fit
  arc_2d/grid_*.png                        all projections side by side

Examples:

python curve/arc_2d.py
python curve/arc_2d.py --book alice_wonderland --model bge-m3 \
  --size 15 --stride 3 --color progression chapter wonder
python curve/arc_2d.py --fit --n-control 10 --lambda 0.1 --show-control
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
from narrative_arc.plot_2d import arc_grid, arc_plot
from narrative_arc.projections import project


def run(book, model, args):
  print(f"\n=== {book} / {model} ===")

  # The series is built and saved first, so what gets projected below is the
  # same object on disk -- the projections are drawings of one series, not
  # separately-derived ones.
  series = W.build(args.data_dir, book, model, args.size, args.stride, args.l2)
  print(f"  {series.stops[-1]} paragraphs -> {len(series)} windows "
        f"(size {args.size}, stride {args.stride})")
  saved = W.save(series, args.output_dir, __file__, args)
  print(f"  series -> {saved}")

  specs = resolve_colors(args.color, series)
  print(f"  coloring by: {', '.join(s.name for s in specs)}")

  projections = project(series.pooled, args.methods, n_components=2,
                        neighbors=args.neighbors, min_dist=args.min_dist,
                        perplexity=args.perplexity, seed=args.seed,
                        pca_fit_rows=cli.pca_rows(args, book, model))

  output_dir = paths.out_dir(args.output_dir, book, model, paths.ARC_2D)
  fit = cli.fit_opts(args)
  win_tag = paths.window_tag(args.size, args.stride, args.l2)
  ftag = fit_tag(fit)
  heading = (f"{book} / {model} -- window {args.size}, stride {args.stride}")

  fits = []
  for proj in projections:
    method_dir = os.path.join(output_dir, proj.tag)
    os.makedirs(method_dir, exist_ok=True)
    np.save(os.path.join(method_dir, f"{proj.tag}_{win_tag}{proj.suffix}.npy"),
            proj.coords)
    proj_fit = cli.fit_projection(
      proj, fit,
      os.path.join(method_dir, f"fit_{proj.tag}_{win_tag}{proj.suffix}{ftag}.npz"))
    fits.append(proj_fit)
    for spec in specs:
      arc_plot(
        proj, spec,
        f"Narrative arc -- {proj.name} (colored by {spec.name})\n{heading}",
        os.path.join(method_dir,
                     f"arc_{proj.tag}_{spec.name}_{win_tag}{proj.suffix}{ftag}.png"),
        fit=proj_fit, show_control=args.show_control,
      )

  if len(projections) > 1 and not args.no_grid:
    for spec in specs:
      arc_grid(
        projections, spec,
        f"Narrative arc (colored by {spec.name}) -- {heading}",
        os.path.join(output_dir, f"grid_{spec.name}_{win_tag}"
                                 f"{cli.grid_suffix(projections)}{ftag}.png"),
        fits=fits, show_control=args.show_control,
      )

  paths.stamp(output_dir, __file__, args, book=book, model=model,
              n_windows=len(series), colorings=[s.name for s in specs],
              projections={p.tag: p.info for p in projections})
  print(f"  -> {output_dir}")


def main():
  parser = argparse.ArgumentParser(
    description="Narrative arc in 2-D: windowed embeddings projected with "
                "PCA, UMAP and t-SNE and joined in reading order.")
  cli.add_arc_args(parser)
  args = parser.parse_args()

  for book, model in cli.targets(args, parser):
    run(book, model, args)
  print("\nDone.")


if __name__ == "__main__":
  main()
