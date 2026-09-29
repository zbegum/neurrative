"""
ctypes bridge to Kirsanov's exact geodesic solver (the vendored C++ in
geodesic_cpp/). The shared library is built from wrapper.cpp on first import and
cached next to it; thereafter import is just a dlopen. Only a C++ compiler is
required -- no pip package, no build step for the user to run.

The one public function, exact_geodesic_paths, is what
geometry.geodesic.exact_paths calls: (vertices, faces, pairs) in, a list of
(m, 3) polylines (or None) out.
"""

import ctypes
import os
import subprocess
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_CPP_DIR = os.path.join(_HERE, "geodesic_cpp")
_SRC = os.path.join(_CPP_DIR, "wrapper.cpp")
_LIB = os.path.join(_CPP_DIR, "libgeodesic_exact" + (".dylib" if sys.platform == "darwin" else ".so"))


class _GeoPaths(ctypes.Structure):
  _fields_ = [
    ("points", ctypes.POINTER(ctypes.c_double)),
    ("counts", ctypes.POINTER(ctypes.c_int)),
    ("npairs", ctypes.c_int),
    ("total_points", ctypes.c_int),
  ]


def _compiler():
  return os.environ.get("CXX", "c++")


def _needs_build():
  if not os.path.exists(_LIB):
    return True
  # Rebuild if any vendored source is newer than the cached library.
  lib_mtime = os.path.getmtime(_LIB)
  for name in os.listdir(_CPP_DIR):
    if name.endswith((".h", ".cpp")):
      if os.path.getmtime(os.path.join(_CPP_DIR, name)) > lib_mtime:
        return True
  return False


def _build():
  """Compile wrapper.cpp + vendored headers into a shared library.

  -std=c++03 because the 2008 sources use std::auto_ptr; -w silences the
  matching deprecation noise. The build is a few source files and takes well
  under a second.
  """
  cmd = [
    _compiler(), "-O2", "-std=c++03", "-w",
    "-fPIC", "-shared",
    "-I", _CPP_DIR,
    _SRC, "-o", _LIB,
  ]
  try:
    subprocess.run(cmd, check=True, capture_output=True, text=True)
  except FileNotFoundError as e:
    raise RuntimeError(
      f"No C++ compiler found ({_compiler()!r}). Set $CXX or install one."
    ) from e
  except subprocess.CalledProcessError as e:
    raise RuntimeError(
      "Failed to build the exact geodesic library:\n" + e.stderr
    ) from e


def _load():
  if _needs_build():
    _build()
  lib = ctypes.CDLL(_LIB)
  lib.geodesic_exact_paths.restype = ctypes.POINTER(_GeoPaths)
  lib.geodesic_exact_paths.argtypes = [
    ctypes.POINTER(ctypes.c_double), ctypes.c_int,   # verts, nverts
    ctypes.POINTER(ctypes.c_uint), ctypes.c_int,     # faces, nfaces
    ctypes.POINTER(ctypes.c_int), ctypes.c_int,      # pairs, npairs
  ]
  lib.geodesic_free.restype = None
  lib.geodesic_free.argtypes = [ctypes.POINTER(_GeoPaths)]
  return lib


_LIB_HANDLE = None


def _lib():
  global _LIB_HANDLE
  if _LIB_HANDLE is None:
    _LIB_HANDLE = _load()
  return _LIB_HANDLE


def exact_geodesic_paths(vertices, faces, pairs):
  """Exact geodesic polylines for (source, target) vertex-index pairs.

  Returns a list of (m, 3) float arrays in R^3, one per pair, running from
  source to target, or None where the solver could not connect the pair (a
  target in a different mesh component). Pairs with source == target return the
  single source point.
  """
  V = np.ascontiguousarray(vertices, dtype=np.float64)
  F = np.ascontiguousarray(faces, dtype=np.uint32)
  P = np.ascontiguousarray(pairs, dtype=np.int32).reshape(-1, 2)

  if len(P) == 0:
    return []

  lib = _lib()
  res = lib.geodesic_exact_paths(
    V.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), len(V),
    F.ctypes.data_as(ctypes.POINTER(ctypes.c_uint)), len(F),
    P.ctypes.data_as(ctypes.POINTER(ctypes.c_int)), len(P),
  )
  if not res:
    raise RuntimeError("exact geodesic solver returned NULL (allocation failed)")

  try:
    r = res.contents
    counts = np.ctypeslib.as_array(r.counts, shape=(r.npairs,)).copy()
    flat = np.ctypeslib.as_array(r.points, shape=(r.total_points * 3,)).copy()
  finally:
    lib.geodesic_free(res)

  pts = flat.reshape(-1, 3)
  paths, off = [], 0
  for c in counts:
    if c == 0:
      paths.append(None)
    else:
      paths.append(pts[off:off + c])
      off += c
  return paths
