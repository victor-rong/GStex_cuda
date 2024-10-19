#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>

__global__ void map_gaussian_to_intersects(
    const int num_points,
    const float2* __restrict__ centers,
    const float2* __restrict__ extents,
    const float* __restrict__ depths,
    const int32_t* __restrict__ cum_tiles_hit,
    const dim3 tile_bounds,
    const unsigned block_width,
    const bool wrapped,
    int64_t* __restrict__ isect_ids,
    int32_t* __restrict__ gaussian_ids
);

__global__ void get_tile_bin_edges(
    const int num_intersects, const int64_t* __restrict__ isect_ids_sorted, int2* __restrict__ tile_bins
);