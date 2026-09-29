"""
Epanechnikov Nadaraya-Watson smoothing of the emotion scores.

Same weighted average as the Gaussian version, but the kernel has compact
support: paragraphs beyond the bandwidth contribute exactly nothing. That makes
it strictly local and MSE-optimal among kernels, at the cost of being able to
run out of neighbours entirely -- where the window is empty the surface is not
low, it is undefined, and the mask drops it.

Because the kernel truncates, its useful bandwidths run larger than the
Gaussian's: h here is a hard radius, not a decay scale.

Example:

python visualization/smooth_epanechnikov_nw.py --model bge-m3
"""

import argparse

from smooth_common import add_common_args, run
from geometry.smoothers import epanechnikov_nw

GRID = {
  "hx": [0.2, 0.4, 0.8, 1.6, 3.2],
  "hy": [0.2, 0.4, 0.8, 1.6, 3.2],
}


def main():
  parser = add_common_args(argparse.ArgumentParser())
  args = parser.parse_args()
  run("epanechnikov_nw", epanechnikov_nw, GRID, args)


if __name__ == "__main__":
  main()
