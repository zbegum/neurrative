"""
Build and save the windowed series without drawing anything.

arc_2d.py and arc_3d.py already do this as their first step; run this on its own
when you only want the pooled series on disk, e.g. to analyse it elsewhere:

    python build_windows.py --all-books --all-models

writes <output-dir>/<book>/<model>/windows/w40_s20/series.npz holding, row-aligned
and in reading order: starts, stops, centers, pooled, scores, emotions, chapters,
center_ids. See narrative_arc/windows.py for what each one is.
"""

import argparse

from narrative_arc import cli
from narrative_arc import windows as W


def main():
  parser = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
  cli.add_target_args(parser)
  cli.add_window_args(parser)
  args = parser.parse_args()

  for book, model in cli.targets(args, parser):
    s = W.build(args.data_dir, book, model, args.size, args.stride, args.l2)
    out = W.save(s, args.output_dir, __file__, args)
    print(f"{book} / {model}: {len(s)} windows x {s.pooled.shape[1]} dims "
          f"(size {s.size}, stride {s.stride}) -> {out}")

  print("\nDone.")


if __name__ == "__main__":
  main()
