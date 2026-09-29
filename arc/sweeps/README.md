# arc/sweeps — how the arc depends on its settings

`sweep.py` redraws the arc across window sizes, B-spline fits, the projections'
own parameters and the embedding models, and writes a metrics CSV beside each
grid (closure, drift, trustworthiness, seed dispersion, fit residual, windows
out of order, end gap). `grid_3d.py` draws the 3-D arc next to its three flat
views for several window sizes, so an apparent self-crossing can be checked.

    python sweeps/sweep.py --all-books --all-models    # run from arc/
    python sweeps/grid_3d.py --all-books --all-models

Details: [../README.md](../README.md#parameter-sweeps).
