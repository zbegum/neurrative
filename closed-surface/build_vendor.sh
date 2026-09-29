#!/usr/bin/env bash
# Build the vendored PoissonRecon binary. macOS/Homebrew, Apple clang.
#
# The vendor's Makefile is left verbatim (see vendor/PROVENANCE.md), so the two
# places it does not fit this machine are worked around from outside:
#
#   1. Its default COMPILER=gcc passes -fopenmp, which Apple clang has no
#      runtime for. COMPILER=clang selects the vendor's own no-OpenMP branch;
#      the solver is still threaded through std::thread.
#   2. Its link rule first builds the bundled PNG/ copy, whose 1990s pngconf.h
#      includes the classic MacOS <fp.h> and fails. Nothing links that build --
#      the binary takes -lpng from the system -- so the link is done here
#      directly instead of through the rule that triggers it.
#
# Header and library locations go through CPATH/LIBRARY_PATH rather than
# CFLAGS/LFLAGS: those are += in the Makefile, and a command-line assignment
# would replace the flags the build needs rather than adding to them.
set -euo pipefail

cd "$(dirname "$0")/vendor/PoissonRecon"

JPEG=$(brew --prefix jpeg-turbo)
PNG=$(brew --prefix libpng)

export CPATH="$JPEG/include:$PNG/include"
export LIBRARY_PATH="$JPEG/lib:$PNG/lib"

mkdir -p Bin/Linux   # the vendor's path for every platform, not a mistake
make Bin/Linux/PoissonRecon.o COMPILER=clang \
  CFLAGS="-Wno-deprecated -std=c++17 -pthread -Wno-invalid-offsetof \
-Wno-dangling-else -Wno-nan-infinity-disabled -O3 -DRELEASE -funroll-loops \
-ffast-math -g"

clang++ -pthread -o Bin/Linux/PoissonRecon Bin/Linux/PoissonRecon.o \
  -lstdc++ -lz -L"$PNG/lib" -lpng -L"$JPEG/lib" -ljpeg

echo "built $(pwd)/Bin/Linux/PoissonRecon"
