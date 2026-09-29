"""
Gaussian Nadaraya-Watson smoothing of the emotion scores.

A weighted average of nearby paragraphs, with the weight falling off as a
Gaussian. The baseline every other smoother is measured against.

Watch the residual plot at the edges: an infinite-support kernel average is
biased at the boundary -- with samples on one side only, the average is pulled
inward and the surface flattens. local_linear exists to fix that.

Example:

python surface/kernel/gaussian.py --model bge-m3
python surface/kernel/gaussian.py --model bge-m3 --emotions wonder
"""

import argparse
import os
import sys

import numpy as np

# The repository root and common/ on the path, wherever this script lives.
_ROOT = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_ROOT, "common")):
  _ROOT = os.path.dirname(_ROOT)
sys.path[1:1] = [_ROOT, os.path.join(_ROOT, "common")]

from smooth_common import add_common_args, run
from geometry.smoothers import gaussian_nw

# Bandwidths are in standard deviations of the standardized coordinates, so the
# ladder means the same thing for either axis and for any book.
GRID = {
  "hx": [0.05, 0.1, 0.2, 0.4, 0.8],
  "hy": [0.05, 0.1, 0.2, 0.4, 0.8],
}


def main():
  parser = add_common_args(argparse.ArgumentParser())
  args = parser.parse_args()
  run("gaussian_nw", gaussian_nw, GRID, args)


if __name__ == "__main__":
  main()
