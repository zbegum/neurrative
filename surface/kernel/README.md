# surface/kernel — kernel smoothing

The height at any point is a weighted average of the paragraphs near it. All
five scripts use the same estimator family (`geometry/smoothers.py`) and the
same protocol (`common/smooth_common.py`): standardized coordinates, the
bandwidth tuned by k-fold cross-validation on a training split, and one score
on a held-out 20% of paragraphs that is never used for fitting or tuning.

| script | method | bandwidth |
|---|---|---|
| `gaussian.py` | Gaussian Nadaraya-Watson: smooth everywhere, but flattens at the edges of the data | `hx`, `hy` by k-fold CV |
| `epanechnikov.py` | Epanechnikov Nadaraya-Watson: finite support, so no influence from far away | `hx`, `hy` by k-fold CV |
| `local_linear.py` | local linear regression: fixes the edge flattening by fitting a slope | `hx`, `hy` by k-fold CV |
| `loess.py` | robust LOESS: a nearest-neighbour span, down-weights outliers | `frac` by k-fold CV |
| `loo.py` | Gaussian with one isotropic `h` in raw PCA units | leave-one-out CV |
| `bandwidth_grid.py` | every smoother across a bandwidth ladder, under- to over-smoothed | sweep |
| `smooth_grid.py` | the smoothers head to head at matched smoothness | sweep |

```
python surface/kernel/gaussian.py --book alice_wonderland --model bge-m3
python surface/kernel/gaussian.py --book alice_wonderland --model bge-m3 --emotions wonder
```

Output: `surface/kernel/output/<book>/<model>/<method>/<variant>/`, with a
`metrics.json` of held-out error against the flat-average baseline.

## Result (Alice, bge-m3, held-out improvement over a flat average)

| emotion | Gaussian | Epanechnikov | local linear | LOESS |
|---|---|---|---|---|
| wonder | 18.5% | 18.5% | 19.0% | 18.4% |
| curiosity | 15.4% | 15.2% | 15.9% | 16.6% |
| danger | 11.2% | 11.5% | 10.6% | 9.4% |
| confusion | 9.6% | 9.4% | 10.0% | 10.0% |
| humor | 6.8% | 6.8% | 7.0% | 6.7% |
| sadness | 4.3% | 4.5% | 4.7% | 2.2% |

On Alice the smoother barely matters; the emotion does. Pride and Prejudice
reverses the order of the emotions (sadness improves by about 20%, wonder by at
most 1%), and there LOESS falls behind: 12.7% on sadness against 19–22% for the
other three, and slightly worse than a flat average on confusion and humor. It
often settles on `frac` 0.8, the edge of its search ladder, so its ladder is
worth widening before reading much into it. `bandwidth_grid.py` and
`smooth_grid.py` are there to look at the other half of that claim: how much
the bandwidth, rather than the kernel, changes the surface.
