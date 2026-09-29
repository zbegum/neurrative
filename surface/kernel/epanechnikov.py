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

python surface/kernel/epanechnikov.py --model bge-m3
"""

import argparse
import os
import sys

# The repository root and common/ on the path, wherever this script lives.
_ROOT = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_ROOT, "common")):
  _ROOT = os.path.dirname(_ROOT)
sys.path[1:1] = [_ROOT, os.path.join(_ROOT, "common")]

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
