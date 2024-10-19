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

__global__ void texture_forward(
    const dim3 tile_bounds,
    const dim3 img_size,
    const int3 texture_info, //{num_charts, num_prob, num_channels}
    const int3* texture_dims, //{h, w, start_index}
    const int32_t* __restrict__ gaussian_ids_sorted,
    const int2* __restrict__ tile_bins,
    const float3* __restrict__ colors,
    const float* __restrict__ opacities,
    const float3* __restrict__ means3d,
    const float3* __restrict__ scales,
    const float glob_scale,
    const float4* __restrict__ quats,
    const float2* __restrict__ uv0s,
    const float3* __restrict__ umaps,
    const float3* __restrict__ vmaps,
    const float* __restrict__ texture,
    const float* __restrict__ viewmat,
    const float* __restrict__ c2w,
    const float4 intrins,
    const int settings,
    const float3& __restrict__ background,
    float* __restrict__ final_Ts,
    int* __restrict__ final_index,
    int* __restrict__ depth_index,
    float3* __restrict__ out_reg_s, // intermediate outputs needed for back propagation, might be faster just to recompute?
    float3* __restrict__ out_img,
    float* __restrict__ out_depth,
    float* __restrict__ out_reg,
    float* __restrict__ out_texture,
    float3* __restrict__ out_normal
);

__global__ void texture_backward(
    const dim3 tile_bounds,
    const dim3 img_size,
    const int3 texture_info, //{num_charts, num_prob, num_channels}
    const int3* texture_dims, //{h, w, start_index}
    const int32_t* __restrict__ gaussian_ids_sorted,
    const int2* __restrict__ tile_bins,
    const float3* __restrict__ colors,
    const float* __restrict__ opacities,
    const float3* __restrict__ means3d,
    const float3* __restrict__ scales,
    const float glob_scale,
    const float4* __restrict__ quats,
    const float2* __restrict__ uv0s,
    const float3* __restrict__ umaps,
    const float3* __restrict__ vmaps,
    const float* __restrict__ texture,
    const float* __restrict__ viewmat,
    const float* __restrict__ c2w,
    const float4 intrins,
    const int settings,
    const float3& __restrict__ background,
    const float* __restrict__ final_Ts,
    const int* __restrict__ final_index,
    int* __restrict__ depth_index,
    const float3* __restrict__ final_s,
    const float3* __restrict__ v_output,
    const float* __restrict__ v_output_depth,
    const float* __restrict__ v_output_reg,
    const float* __restrict__ v_output_alpha,
    const float* __restrict__ v_output_texture,
    const float3* __restrict__ v_output_normal,
    float3* __restrict__ v_rgb,
    float* __restrict__ v_opacity,
    float3* __restrict__ v_mean,
    float3* __restrict__ v_scale,
    float4* __restrict__ v_quat,
    float2* __restrict__ v_uv0,
    float3* __restrict__ v_umap,
    float3* __restrict__ v_vmap,
    float* __restrict__ v_texture
);

std::tuple<
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor
>
texture_forward_tensor(
    const std::tuple<int, int, int> tile_bounds,
    const std::tuple<int, int, int> block,
    const std::tuple<int, int, int> img_size,
    const std::tuple<int, int, int> texture_info,
    const torch::Tensor &texture_dims,
    const torch::Tensor &gaussian_ids_sorted,
    const torch::Tensor &tile_bins,
    const torch::Tensor &colors,
    const torch::Tensor &opacities,
    const torch::Tensor &means,
    const torch::Tensor &scales,
    const float glob_scale,
    const torch::Tensor &quats,
    const torch::Tensor &uv0,
    const torch::Tensor &umap,
    const torch::Tensor &vmap,
    const torch::Tensor &texture,
    const torch::Tensor &viewmat,
    const torch::Tensor &c2w,
    const float fx,
    const float fy,
    const float cx,
    const float cy,
    const int settings,
    const torch::Tensor &background
);

std::
    tuple<
        torch::Tensor, // dL_dcolors
        torch::Tensor, // dL_dopacity
        torch::Tensor, // dL_dmeans3d
        torch::Tensor, // dL_dscales
        torch::Tensor, // dL_dquats
        torch::Tensor, // dL_duv0
        torch::Tensor, // dL_dumap
        torch::Tensor, // dL_dvmap
        torch::Tensor  // dL_dtexture
        >
    texture_backward_tensor(
        const unsigned img_height,
        const unsigned img_width,
        const unsigned block_width,
        const std::tuple<int, int, int> texture_info,
        const torch::Tensor &texture_dims,
        const torch::Tensor &gaussian_ids_sorted,
        const torch::Tensor &tile_bins,
        const torch::Tensor &colors,
        const torch::Tensor &opacities,
        const torch::Tensor &means,
        const torch::Tensor &scales,
        const float glob_scale,
        const torch::Tensor &quats,
        const torch::Tensor &uv0,
        const torch::Tensor &umap,
        const torch::Tensor &vmap,
        const torch::Tensor &texture,
        const torch::Tensor &viewmat,
        const torch::Tensor &c2w,
        const float fx,
        const float fy,
        const float cx,
        const float cy,
        const int settings,
        const torch::Tensor &background,
        const torch::Tensor &final_Ts,
        const torch::Tensor &final_idx,
        const torch::Tensor &depth_idx,
        const torch::Tensor &final_s,
        const torch::Tensor &v_output, // dL_dout_color
        const torch::Tensor &v_output_depth,
        const torch::Tensor &v_output_reg,
        const torch::Tensor &v_output_alpha, // dL_dout_alpha
        const torch::Tensor &v_output_texture, // dL_dout_texture
        const torch::Tensor &v_output_normal
    );