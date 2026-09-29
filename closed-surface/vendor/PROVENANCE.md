# Vendored: PoissonRecon

    https://github.com/mkazhdan/PoissonRecon
    commit 262b0f539d404057d1f36e1adc07fc9388678899  ("Version 8.76", 2026-04-29)
    MIT License (see PoissonRecon/LICENSE)

Cloned verbatim, shallow, and **not modified** -- `git diff` inside it should
stay empty. Unlike `b-surface/vendor/`, this is code we run rather than a
specification we ported: it is MIT licensed, it is C++ that builds here, and
reimplementing an octree multigrid solver to avoid shelling out to a binary
would be a much worse trade than the one that folder makes.

Papers this implements, for reading a parameter's meaning rather than its flag:

    Kazhdan, Bolitho, Hoppe. Poisson Surface Reconstruction. SGP 2006.
    Kazhdan, Hoppe. Screened Poisson Surface Reconstruction. TOG 32(3), 2013.
                    -- this is where `--pointWeight` comes from.

## What we build, and what we do not

Only the `PoissonRecon` binary; `build_vendor.sh` says how and why the vendor's
own `make poissonrecon` does not get there unassisted on macOS. The rest of the
suite (`SSDRecon`, `PointInterpolant`, `SurfaceTrimmer`, the client/server pair,
the image tools) is left unbuilt but not deleted, so the tree matches upstream
and a later need can just build it.

`SurfaceTrimmer` is the one worth knowing about: it cuts a reconstruction back
to where the samples actually supported it. Nothing here calls it, because the
cloud `poisson.py` builds is already closed by construction -- floor and walls
included -- so there is no unsupported extrapolation to trim. If the floor and
walls are ever dropped in favour of reconstructing the bare paragraph sheet,
that is the tool the result will need.

## The binary is not committed

`Bin/Linux/PoissonRecon` is a 22 MB build artifact and is gitignored along with
the object file. Run `closed-surface/build_vendor.sh` after a fresh clone;
`poisson.py` raises with that instruction if the binary is missing.
