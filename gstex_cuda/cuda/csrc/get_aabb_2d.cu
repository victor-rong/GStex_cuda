#include "get_aabb_2d.cuh"
#include "helpers.cuh"
#include <algorithm>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <iostream>
#include <cuda_fp16.h>

namespace cg = cooperative_groups;

__global__ void get_aabb_2d_kernel(
    const int num_points,
    const float3* __restrict__ means,
    const float3* __restrict__ scales,
    const float glob_scale,
    const float4* __restrict__ quats,
    const float* __restrict__ viewmat,
    const float4 intrins,
    float2* __restrict__ centers,
    float2* __restrict__ extents
) {
    unsigned idx = cg::this_grid().thread_rank();
    if (idx >= num_points)
        return;
    const float t_near = 0.01f;
    const float t_far = 1000.0f;
    float fx = intrins.x;
    float fy = intrins.y;
    float cx = intrins.z;
    float cy = intrins.w;

    float3 point = means[idx];
    float3 p_view;
    bool clipped = clip_near_plane(point, viewmat, p_view, t_near);
    float2 xy = project_pix({fx, fy}, p_view, {cx, cy});

    float4 quat = quats[idx];

    glm::mat3 R = quat_to_rotmat(quat);
    const float ell = 3.0f * glob_scale;
    const float3 scale = scales[idx];
    const float3 ax1 = {R[0][0], R[0][1], R[0][2]};
    const float3 ax2 = {R[1][0], R[1][1], R[1][2]};
    const float3 ax3 = {R[2][0], R[2][1], R[2][2]};

    const float3 corner00 = {
        point.x + ell * scale.x * ax1.x + ell * scale.y * ax2.x,
        point.y + ell * scale.x * ax1.y + ell * scale.y * ax2.y,
        point.z + ell * scale.x * ax1.z + ell * scale.y * ax2.z
    };
    const float3 corner01 = {
        point.x + ell * scale.x * ax1.x - ell * scale.y * ax2.x,
        point.y + ell * scale.x * ax1.y - ell * scale.y * ax2.y,
        point.z + ell * scale.x * ax1.z - ell * scale.y * ax2.z
    };
    const float3 corner10 = {
        point.x - ell * scale.x * ax1.x + ell * scale.y * ax2.x,
        point.y - ell * scale.x * ax1.y + ell * scale.y * ax2.y,
        point.z - ell * scale.x * ax1.z + ell * scale.y * ax2.z
    };
    const float3 corner11 = {
        point.x - ell * scale.x * ax1.x - ell * scale.y * ax2.x,
        point.y - ell * scale.x * ax1.y - ell * scale.y * ax2.y,
        point.z - ell * scale.x * ax1.z - ell * scale.y * ax2.z
    };
    float2 pix00 = clip_and_project(corner00, viewmat, t_near, intrins);
    float2 pix01 = clip_and_project(corner01, viewmat, t_near, intrins);
    float2 pix10 = clip_and_project(corner10, viewmat, t_near, intrins);
    float2 pix11 = clip_and_project(corner11, viewmat, t_near, intrins);

    float2 maxval = {
        max(max(pix00.x, pix01.x), max(pix10.x, pix11.x)),
        max(max(pix00.y, pix01.y), max(pix10.y, pix11.y))
    };
    float2 minval = {
        min(min(pix00.x, pix01.x), min(pix10.x, pix11.x)),
        min(min(pix00.y, pix01.y), min(pix10.y, pix11.y))
    };
    float2 center = {0.5f * (maxval.x + minval.x), 0.5f * (maxval.y + minval.y)};
    float2 extent = {0.5f * (maxval.x - minval.x), 0.5f * (maxval.y - minval.y)};
    if (clipped) {
        centers[idx] = xy;
        extents[idx] = {0.f, 0.f};
    }
    else {
        centers[idx] = center;
        extents[idx] = extent;
    }
}

std::tuple<torch::Tensor, torch::Tensor> get_aabb_2d_tensor(
    const torch::Tensor &means,
    const torch::Tensor &scales,
    const float glob_scale,
    const torch::Tensor &quats,
    const torch::Tensor &viewmat,
    const float fx,
    const float fy,
    const float cx,
    const float cy
) {
    CHECK_INPUT(means);
    CHECK_INPUT(scales);
    CHECK_INPUT(quats);
    CHECK_INPUT(viewmat);

    float4 intrins = {fx, fy, cx, cy};
    const int num_points = means.size(0);

    torch::Tensor centers = torch::zeros({num_points, 2}, means.options());
    torch::Tensor extents = torch::zeros({num_points, 2}, means.options());

    int blocks = (num_points + N_THREADS - 1) / N_THREADS;
    get_aabb_2d_kernel<<<blocks, N_THREADS>>>(
        num_points,
        (float3 *)means.contiguous().data_ptr<float>(),
        (float3 *)scales.contiguous().data_ptr<float>(),
        glob_scale,
        (float4 *)quats.contiguous().data_ptr<float>(),
        viewmat.contiguous().data_ptr<float>(),
        intrins,
        (float2 *)centers.contiguous().data_ptr<float>(),
        (float2 *)extents.contiguous().data_ptr<float>()
    );
    return std::make_tuple(centers, extents);
}