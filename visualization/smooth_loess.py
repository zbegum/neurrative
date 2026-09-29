"""
Robust LOESS smoothing of the emotion scores.

Local polynomial regression on a k-nearest-neighbour window, so the
neighbourhood adapts to local density: tight where paragraphs crowd together,
wide out in the sparse regions. Tricube weights fall to zero at the window edge.

The robustness iterations are the point. Our scores are LLM annotations, and a
handful of them are wrong -- a plain least-squares fit lets those outliers drag
the surface toward themselves. Bisquare reweighting notices which samples the
fit cannot explain and discounts them, so one mislabelled paragraph does not
raise a hill.

Compare its residual histogram to the other three: LOESS should leave a few
large residuals standing rather than smearing them across the neighbourhood.

Example:

python visualization/smooth_loess.py --model bge-m3
python visualization/smooth_loess.py --model bge-m3 --emotions wonder
"""

import argparse

from smooth_common import add_common_args, run
from geometry.smoothers import loess

# frac is the neighbourhood as a fraction of the samples -- the LOESS analogue
# of a bandwidth, and what cross-validation tunes here.
GRID = {
  "frac": [0.05, 0.1, 0.2, 0.4, 0.8],
  "degree": [1, 2],
}


def main():
  parser = add_common_args(argparse.ArgumentParser())
  parser.add_argument("--iterations", default=2, type=int,
                      help="Bisquare reweighting rounds. 0 disables robustness.")
  args = parser.parse_args()

  grid = dict(GRID, iterations=[args.iterations])
  run("loess", loess, grid, args)


if __name__ == "__main__":
  main()
