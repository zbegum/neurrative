"""
Local linear (and quadratic) kernel regression of the emotion scores.

Instead of averaging the neighbours, fit a plane through them at every query
point and read off its value there. The payoff is at the boundary: where a
kernel average sees samples on one side only and flattens toward them, a tilted
local fit carries the trend off the edge of the data. With half our grid masked
as empty latent space, most of this surface is boundary.

degree is tuned alongside the bandwidths: 1 fits a plane, 2 a paraboloid, which
tracks a curved ridge better but needs more samples in the window to stay stable.

Example:

python surface/kernel/local_linear.py --model bge-m3
python surface/kernel/local_linear.py --model bge-m3 --emotions wonder
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
from geometry.smoothers import local_linear

GRID = {
  "hx": [0.1, 0.2, 0.4, 0.8, 1.6],
  "hy": [0.1, 0.2, 0.4, 0.8, 1.6],
  "degree": [1, 2],
}


def main():
  parser = add_common_args(argparse.ArgumentParser())
  args = parser.parse_args()
  run("local_linear", local_linear, GRID, args)


if __name__ == "__main__":
  main()
