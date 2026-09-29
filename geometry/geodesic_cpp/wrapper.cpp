// C ABI shim over Danil Kirsanov's exact geodesic library (MMP algorithm,
// geodesic_algorithm_exact.h). Compiled to a shared library and driven from
// Python via ctypes -- the geodesic computation itself is entirely upstream
// code; this file only marshals arrays across the boundary.
//
// The one entry point, geodesic_exact_paths, computes exact source->target
// geodesic polylines for a batch of vertex-index pairs on a single mesh. It
// propagates the distance field once per *distinct source* (not once per pair),
// mirroring the batching the pure-Python Dijkstra baseline already did, then
// traces each target back. Results are returned as one flat coordinate buffer
// plus a per-pair point count; the caller frees both with geodesic_free.

#include <vector>
#include <map>

#include "geodesic_algorithm_exact.h"

extern "C" {

// A batch of geodesic paths. `points` is the concatenation of every path's
// (x, y, z) triples, in the pair order given by the caller. `counts[i]` is the
// number of points in path i; 0 marks a pair the solver could not connect
// (e.g. the target lies in a different mesh component). `total_points` is the
// sum of counts, i.e. points holds 3*total_points doubles.
struct GeoPaths {
	double*   points;
	int*      counts;
	int       npairs;
	int       total_points;
};

// verts:  nverts*3 doubles, row-major (x, y, z).
// faces:  nfaces*3 unsigned, vertex indices.
// pairs:  npairs*2 ints, (source_vertex, target_vertex).
// Returns a heap-allocated GeoPaths (free with geodesic_free), or NULL on a
// bad allocation.
GeoPaths* geodesic_exact_paths(const double*   verts, int nverts,
                               const unsigned* faces, int nfaces,
                               const int*      pairs, int npairs)
{
	// Upstream initialize_mesh_data takes std::vector inputs by reference.
	std::vector<double>   points(verts, verts + (size_t)nverts * 3);
	std::vector<unsigned> tris(faces, faces + (size_t)nfaces * 3);

	geodesic::Mesh mesh;
	mesh.initialize_mesh_data(points, tris);

	geodesic::GeodesicAlgorithmExact algorithm(&mesh);

	// Group pair indices by their source vertex so each source is propagated
	// exactly once; every pair sharing that source is then traced back cheaply.
	std::map<int, std::vector<int> > by_source;
	for (int i = 0; i < npairs; ++i) {
		by_source[pairs[2 * i]].push_back(i);
	}

	// Per-pair traced polylines, filled out of order and reassembled below.
	std::vector<std::vector<geodesic::SurfacePoint> > traced(npairs);

	for (std::map<int, std::vector<int> >::iterator it = by_source.begin();
	     it != by_source.end(); ++it) {
		int src = it->first;
		std::vector<int>& members = it->second;

		geodesic::SurfacePoint source(&mesh.vertices()[src]);
		std::vector<geodesic::SurfacePoint> all_sources(1, source);

		// Stop propagation once every target for this source is covered, rather
		// than blanketing the whole mesh.
		std::vector<geodesic::SurfacePoint> stop_points;
		stop_points.reserve(members.size());
		for (size_t k = 0; k < members.size(); ++k) {
			int dst = pairs[2 * members[k] + 1];
			stop_points.push_back(geodesic::SurfacePoint(&mesh.vertices()[dst]));
		}

		algorithm.propagate(all_sources, geodesic::GEODESIC_INF, &stop_points);

		for (size_t k = 0; k < members.size(); ++k) {
			int pi  = members[k];
			int dst = pairs[2 * pi + 1];

			if (dst == src) {                       // degenerate: single point
				traced[pi].push_back(source);
				continue;
			}

			geodesic::SurfacePoint target(&mesh.vertices()[dst]);

			// If propagation never reached the target (disconnected component),
			// leave the path empty -> reported as a failure, never a fake line.
			double d = geodesic::GEODESIC_INF;
			algorithm.best_source(target, d);
			if (d >= geodesic::GEODESIC_INF) {
				continue;
			}

			algorithm.trace_back(target, traced[pi]);   // target -> ... -> source
		}
	}

	GeoPaths* out = new (std::nothrow) GeoPaths();
	if (!out) return 0;
	out->npairs = npairs;

	int total = 0;
	for (int i = 0; i < npairs; ++i) total += (int)traced[i].size();
	out->total_points = total;

	out->counts = new (std::nothrow) int[npairs > 0 ? npairs : 1];
	out->points = new (std::nothrow) double[(total > 0 ? total : 1) * 3];
	if (!out->counts || !out->points) {
		delete[] out->counts;
		delete[] out->points;
		delete out;
		return 0;
	}

	int w = 0;
	for (int i = 0; i < npairs; ++i) {
		std::vector<geodesic::SurfacePoint>& path = traced[i];
		out->counts[i] = (int)path.size();
		// trace_back yields target -> source; emit reversed so every path runs
		// source -> target, matching the (src, dst) the caller asked for.
		for (size_t j = path.size(); j-- > 0; ) {
			out->points[3 * w + 0] = path[j].x();
			out->points[3 * w + 1] = path[j].y();
			out->points[3 * w + 2] = path[j].z();
			++w;
		}
	}

	return out;
}

void geodesic_free(GeoPaths* p)
{
	if (!p) return;
	delete[] p->points;
	delete[] p->counts;
	delete p;
}

} // extern "C"
