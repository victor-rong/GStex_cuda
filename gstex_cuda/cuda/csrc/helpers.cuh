#include "config.h"
#include <cuda_runtime.h>
#include "third_party/glm/glm/glm.hpp"
#include "third_party/glm/glm/gtc/type_ptr.hpp"
#include <cuda_fp16.h>
#include <iostream>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

inline __device__ void warpSum4(float4& val, cg::thread_block_tile<32>& tile){
    val.x = cg::reduce(tile, val.x, cg::plus<float>());
    val.y = cg::reduce(tile, val.y, cg::plus<float>());
    val.z = cg::reduce(tile, val.z, cg::plus<float>());
    val.w = cg::reduce(tile, val.w, cg::plus<float>());
}

inline __device__ void warpSum3(float3& val, cg::thread_block_tile<32>& tile){
    val.x = cg::reduce(tile, val.x, cg::plus<float>());
    val.y = cg::reduce(tile, val.y, cg::plus<float>());
    val.z = cg::reduce(tile, val.z, cg::plus<float>());
}

inline __device__ void warpSum2(float2& val, cg::thread_block_tile<32>& tile){
    val.x = cg::reduce(tile, val.x, cg::plus<float>());
    val.y = cg::reduce(tile, val.y, cg::plus<float>());
}

inline __device__ void warpSum(float& val, cg::thread_block_tile<32>& tile){
    val = cg::reduce(tile, val, cg::plus<float>());
}

inline __device__ float dot(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline __device__ void get_bbox(
    const float2 center,
    const float2 dims,
    const dim3 img_size,
    int2 &bb_min,
    int2 &bb_max
) {
    // get bounding box with center and dims, within bounds
    // bounding box coords returned in tile coords, inclusive min, exclusive max
    // clamp between 0 and tile bounds
    bb_min.x = min(max(0, (int)(center.x - dims.x)), img_size.x);
    bb_max.x = min(max(0, (int)(center.x + dims.x + 1)), img_size.x);
    bb_min.y = min(max(0, (int)(center.y - dims.y)), img_size.y);
    bb_max.y = min(max(0, (int)(center.y + dims.y + 1)), img_size.y);
}

inline __device__ void get_bbox_wrapped(
    const float2 center,
    const float2 dims,
    const dim3 img_size,
    int2 &bb_min,
    int2 &bb_max
) {
    // get bounding box with center and dims, within bounds
    // bounding box coords returned in tile coords, inclusive min, exclusive max
    // clamp between 0 and tile bounds
    bb_min.x = (int)(center.x - dims.x);
    if (bb_min.x <= 0) {
        bb_min.x -= 1;
    }
    bb_max.x = (int)(center.x + dims.x + 1);
    bb_min.y = (int)(center.y - dims.y);
    if (bb_min.y <= 0) {
        bb_min.y -= 1;
    }
    bb_max.y = (int)(center.y + dims.y + 1);
}

inline __device__ void get_tile_bbox(
    const float2 pix_center,
    const float2 pix_extent,
    const dim3 tile_bounds,
    int2 &tile_min,
    int2 &tile_max,
    const int block_size
) {
    // gets gaussian dimensions in tile space, i.e. the span of a gaussian in
    // tile_grid (image divided into tiles)
    float2 tile_center = {
        pix_center.x / (float)block_size, pix_center.y / (float)block_size
    };
    float2 tile_extent = {
        pix_extent.x / (float)block_size, pix_extent.y / (float)block_size
    };
    get_bbox(tile_center, tile_extent, tile_bounds, tile_min, tile_max);
}

inline __device__ void get_tile_bbox_wrapped(
    const float2 pix_center,
    const float2 pix_extent,
    const dim3 tile_bounds,
    int2 &tile_min,
    int2 &tile_max,
    const int block_size
) {
    // gets gaussian dimensions in tile space, i.e. the span of a gaussian in
    // tile_grid (image divided into tiles)
    float2 tile_center = {
        pix_center.x / (float)block_size, pix_center.y / (float)block_size
    };
    float2 tile_extent = {
        pix_extent.x / (float)block_size, pix_extent.y / (float)block_size
    };
    get_bbox_wrapped(tile_center, tile_extent, tile_bounds, tile_min, tile_max);
}

// helper for applying R^T * p for a ROW MAJOR 4x3 matrix [R, t], ignoring t
inline __device__ float3 transform_4x3_rot_only_transposed(const float *mat, const float3 p) {
    float3 out = {
        mat[0] * p.x + mat[4] * p.y + mat[8] * p.z,
        mat[1] * p.x + mat[5] * p.y + mat[9] * p.z,
        mat[2] * p.x + mat[6] * p.y + mat[10] * p.z,
    };
    return out;
}

// helper for applying R * p + T, expect mat to be ROW MAJOR
inline __device__ float3 transform_4x3(const float *mat, const float3 p) {
    float3 out = {
        mat[0] * p.x + mat[1] * p.y + mat[2] * p.z + mat[3],
        mat[4] * p.x + mat[5] * p.y + mat[6] * p.z + mat[7],
        mat[8] * p.x + mat[9] * p.y + mat[10] * p.z + mat[11],
    };
    return out;
}

// helper to apply 4x4 transform to 3d vector, return homo coords
// expects mat to be ROW MAJOR
inline __device__ float4 transform_4x4(const float *mat, const float3 p) {
    float4 out = {
        mat[0] * p.x + mat[1] * p.y + mat[2] * p.z + mat[3],
        mat[4] * p.x + mat[5] * p.y + mat[6] * p.z + mat[7],
        mat[8] * p.x + mat[9] * p.y + mat[10] * p.z + mat[11],
        mat[12] * p.x + mat[13] * p.y + mat[14] * p.z + mat[15],
    };
    return out;
}

inline __device__ float2 project_pix(
    const float2 fxfy, const float3 p_view, const float2 pp
) {
    float rw = 1.f / (p_view.z + 1e-6f);
    float2 p_proj = { p_view.x * rw, p_view.y * rw };
    float2 p_pix = { p_proj.x * fxfy.x + pp.x, p_proj.y * fxfy.y + pp.y };
    return p_pix;
}

// given v_xy_pix, get v_xyz
inline __device__ float3 project_pix_vjp(
    const float2 fxfy, const float3 p_view, const float2 v_xy
) {
    float rw = 1.f / (p_view.z + 1e-6f);
    float2 v_proj = { fxfy.x * v_xy.x, fxfy.y * v_xy.y };
    float3 v_view = {
        v_proj.x * rw, v_proj.y * rw, -(v_proj.x * p_view.x + v_proj.y * p_view.y) * rw * rw
    };
    return v_view;
}

inline __device__ glm::mat3 quat_to_rotmat(const float4 quat) {
    // quat to rotation matrix
    float w = quat.x;
    float x = quat.y;
    float y = quat.z;
    float z = quat.w;

    // glm matrices are column-major
    return glm::mat3(
        1.f - 2.f * (y * y + z * z),
        2.f * (x * y + w * z),
        2.f * (x * z - w * y),
        2.f * (x * y - w * z),
        1.f - 2.f * (x * x + z * z),
        2.f * (y * z + w * x),
        2.f * (x * z + w * y),
        2.f * (y * z - w * x),
        1.f - 2.f * (x * x + y * y)
    );
}

inline __device__ float4
quat_to_rotmat_vjp(const float4 quat, const glm::mat3 v_R) {
    float w = quat.x;
    float x = quat.y;
    float y = quat.z;
    float z = quat.w;

    float4 v_quat;
    // v_R is COLUMN MAJOR
    // w element stored in x field
    v_quat.x =
        2.f * (
                  // v_quat.w = 2.f * (
                  x * (v_R[1][2] - v_R[2][1]) + y * (v_R[2][0] - v_R[0][2]) +
                  z * (v_R[0][1] - v_R[1][0])
              );
    // x element in y field
    v_quat.y =
        2.f *
        (
            // v_quat.x = 2.f * (
            -2.f * x * (v_R[1][1] + v_R[2][2]) + y * (v_R[0][1] + v_R[1][0]) +
            z * (v_R[0][2] + v_R[2][0]) + w * (v_R[1][2] - v_R[2][1])
        );
    // y element in z field
    v_quat.z =
        2.f *
        (
            // v_quat.y = 2.f * (
            x * (v_R[0][1] + v_R[1][0]) - 2.f * y * (v_R[0][0] + v_R[2][2]) +
            z * (v_R[1][2] + v_R[2][1]) + w * (v_R[2][0] - v_R[0][2])
        );
    // z element in w field
    v_quat.w =
        2.f *
        (
            // v_quat.z = 2.f * (
            x * (v_R[0][2] + v_R[2][0]) + y * (v_R[1][2] + v_R[2][1]) -
            2.f * z * (v_R[0][0] + v_R[1][1]) + w * (v_R[0][1] - v_R[1][0])
        );
    return v_quat;
}

inline __device__ glm::mat3
scale_to_mat(const float3 scale, const float glob_scale) {
    glm::mat3 S = glm::mat3(1.f);
    S[0][0] = glob_scale * scale.x;
    S[1][1] = glob_scale * scale.y;
    S[2][2] = glob_scale * scale.z;
    return S;
}

// device helper for culling near points
inline __device__ bool clip_near_plane(
    const float3 p, const float *viewmat, float3 &p_view, float thresh
) {
    p_view = transform_4x3(viewmat, p);
    if (p_view.z <= thresh) {
        return true;
    }
    return false;
}

inline __device__ float2 clip_and_project(
    const float3 p, const float *viewmat, float t_near, float4 intrins
) {
    float3 p_view;
    clip_near_plane(p, viewmat, p_view, t_near);
    p_view.z = max(p_view.z, t_near);
    return project_pix({intrins.x, intrins.y}, p_view, {intrins.z, intrins.w});
}

inline __device__ void distortion(const float val, const float weight, float3& val_s, float& err) {
    err += weight * (val * val * val_s.x + val_s.z - 2.f * val * val_s.y);
    val_s.x += weight;
    val_s.y += weight * val;
    val_s.z += weight * val * val;
}

inline __device__ void distortion_vjp(const float val, const float weight, const float3 val_s, const float v_err, float& v_val, float& v_weight) {
    v_val += 2.f * (weight * val * val_s.x - weight * val_s.y) * v_err;
    v_weight += (val * val * val_s.x - 2.f * val * val_s.y + val_s.z) * v_err;
}