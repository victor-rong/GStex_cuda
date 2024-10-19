#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <math.h>
#include <torch/extension.h>
#include <tuple>
#include <c10/cuda/CUDAGuard.h>
#include "bindings.h"

__global__ void texture_edit(
    const dim3 tile_bounds,
    const dim3 img_size,
    const int3 texture_info, //{num_charts, num_prob, num_channels}
    const int3* texture_dims, //{h, w, start_index}
    const float3* __restrict__ updated_img,
    const float* __restrict__ updated_alpha,
    const float* __restrict__ depth_lower,
    const float* __restrict__ depth_upper,
    const int32_t* __restrict__ gaussian_ids_sorted,
    const int2* __restrict__ tile_bins,
    const float* __restrict__ opacities,
    const float3* __restrict__ means3d,
    const float3* __restrict__ scales,
    const float glob_scale,
    const float4* __restrict__ quats,
    const float2* __restrict__ uv0s,
    const float3* __restrict__ umaps,
    const float3* __restrict__ vmaps,
    const float* __restrict__ viewmat,
    const float* __restrict__ c2w,
    const float4 intrins,
    const int settings,
    const float3& __restrict__ background,
    float* __restrict__ updated_texture
);

torch::Tensor
texture_edit_tensor(
    const std::tuple<int, int, int> tile_bounds,
    const std::tuple<int, int, int> block,
    const std::tuple<int, int, int> img_size,
    const std::tuple<int, int, int> texture_info,
    const int texture_total_size,
    const torch::Tensor &texture_dims,
    const torch::Tensor &updated_img,
    const torch::Tensor &updated_alpha,
    const torch::Tensor &depth_lower,
    const torch::Tensor &depth_upper,
    const torch::Tensor &gaussian_ids_sorted,
    const torch::Tensor &tile_bins,
    const torch::Tensor &opacities,
    const torch::Tensor &means,
    const torch::Tensor &scales,
    const float glob_scale,
    const torch::Tensor &quats,
    const torch::Tensor &uv0,
    const torch::Tensor &umap,
    const torch::Tensor &vmap,
    const torch::Tensor &viewmat,
    const torch::Tensor &c2w,
    const float fx,
    const float fy,
    const float cx,
    const float cy,
    const int settings,
    const torch::Tensor &background
);