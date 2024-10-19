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

#include "helpers.cuh"
#include <algorithm>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <iostream>
#include <cuda_fp16.h>

inline __device__ void get_uv(const float2 uv0, const float3 umap, const float3 vmap,
    const float3 diff, float2& uv
) {
    uv.x = uv0.x + dot(umap, diff);
    uv.y = uv0.y + dot(vmap, diff);
}

inline __device__ void wrap_coord(float2& coord) {
    if (coord.x <= 0.0f) {
        coord.x = 1.0f + coord.x;
    }
    if (coord.y <= 0.0f) {
        coord.y = 1.0f + coord.y;
    }
    coord.x = fmod(fmod(coord.x, 1.0f) + 1.0f, 1.0f);
    coord.y = fmod(fmod(coord.y, 1.0f) + 1.0f, 1.0f);
}

inline __device__ void clamp_coord(float2& coord) {
    float eps = 0.f;
    if (coord.x <= eps) {
        coord.x = eps;
    }
    if (coord.x >= 1.f - eps) {
        coord.x = 1.f - eps;
    }
    if (coord.y <= eps) {
        coord.y = eps;
    }
    if (coord.y >= 1.f - eps) {
        coord.y = 1.f - eps;
    }
}

inline __device__ void fix_coord(float2& coord, const bool replicate_pad) {
    if (replicate_pad) {
        clamp_coord(coord);
    }
    else {
        wrap_coord(coord);
    }
}

inline __device__ void get_uv_vjp(
    const float2 uv0, const float3 umap, const float3 vmap,
    const float3 diff, const float2 v_uv,
    float2& v_uv0, float3& v_umap, float3& v_vmap, float3& v_diff
) {
    v_uv0.x += v_uv.x;
    v_uv0.y += v_uv.y;

    v_umap.x += diff.x * v_uv.x;
    v_umap.y += diff.y * v_uv.x;
    v_umap.z += diff.z * v_uv.x;
    v_vmap.x += diff.x * v_uv.y;
    v_vmap.y += diff.y * v_uv.y;
    v_vmap.z += diff.z * v_uv.y;
    // don't backprop to position
    v_diff.x += (umap.x * v_uv.x + vmap.x * v_uv.y);
    v_diff.y += (umap.y * v_uv.x + vmap.y * v_uv.y);
    v_diff.z += (umap.z * v_uv.x + vmap.z * v_uv.y);
}

inline __device__ void get_uv_ls(const float2 uv0, const float3 umap, const float3 vmap, const float2 coord,
    const float3 ax1, const float3 ax2, float2& als
) {
    float2 dc1 = {dot(umap, ax1), dot(vmap, ax1)};
    float2 dc2 = {dot(umap, ax2), dot(vmap, ax2)};

    float2 fd = {coord.x - uv0.x, coord.y - uv0.y};

    float det = dc1.x * dc2.y - dc1.y * dc2.x;
    float al1 = (fd.x * dc2.y - fd.y * dc2.x) / det;
    float al2 = (fd.y * dc1.x - fd.x * dc1.y) / det;
    als = {al1, al2};
}

inline __device__ void get_uv_ls_vjp(const float2 uv0, const float3 umap, const float3 vmap, const float2 coord,
    const float3 ax1, const float3 ax2, const float2 v_als, float2& v_uv0, float3& v_umap, float3& v_vmap
) {
    float2 dc1 = {dot(umap, ax1), dot(vmap, ax1)};
    float2 dc2 = {dot(umap, ax2), dot(vmap, ax2)};

    float2 fd = {coord.x - uv0.x, coord.y - uv0.y};

    float det = dc1.x * dc2.y - dc1.y * dc2.x;
    float al1 = (fd.x * dc2.y - fd.y * dc2.x) / det;
    float al2 = (fd.y * dc1.x - fd.x * dc1.y) / det;

    float v_l1 = v_als.x;
    float v_l2 = v_als.y;

    float v_det = -(al1 * v_l1 + al2 * v_l2) / det;

    float2 v_fd = {
        (dc2.y * v_l1 - dc1.y * v_l2) / det,
        (dc1.x * v_l2 - dc2.x * v_l1) / det
    };

    float2 v_dc1 = {
        dc2.y * v_det + fd.y * v_l2 / det,
        -dc2.x * v_det - fd.x * v_l2 / det
    };

    float2 v_dc2 = {
        -dc1.y * v_det - fd.y * v_l1 / det,
        dc1.x * v_det + fd.x * v_l1 / det
    };

    v_uv0.x -= v_fd.x;
    v_uv0.y -= v_fd.y;

    v_umap.x += (ax1.x * v_dc1.x + ax2.x * v_dc2.x);
    v_umap.y += (ax1.y * v_dc1.x + ax2.y * v_dc2.x);
    v_umap.z += (ax1.z * v_dc1.x + ax2.z * v_dc2.x);
    v_vmap.x += (ax1.x * v_dc1.y + ax2.x * v_dc2.y);
    v_vmap.y += (ax1.y * v_dc1.y + ax2.y * v_dc2.y);
    v_vmap.z += (ax1.z * v_dc1.y + ax2.z * v_dc2.y);
}

inline __device__ void get_uv_alpha(const float2 als, const float2 scales, float& uv_alpha) {
    float uv_sigma = 0.5f * (als.x * als.x / (scales.x * scales.x) + als.y * als.y / (scales.y * scales.y));
    uv_alpha = expf(-uv_sigma);
}

inline __device__ void get_uv_alpha_vjp(
    const float2 als, const float2 scales, const float v_uv_alpha, float2& v_als
) {
    float uv_sigma = 0.5f * (als.x * als.x / (scales.x * scales.x) + als.y * als.y / (scales.y * scales.y));

    const float v_uv_sigma = -expf(-uv_sigma) * v_uv_alpha;

    v_als.x += (als.x / (scales.x * scales.x) * v_uv_sigma);
    v_als.y += (als.y / (scales.y * scales.y) * v_uv_sigma);
}

inline __device__ int fix_int_coord(int val, const int maxval, const bool replicate_pad) {
    if (replicate_pad) {
        return min(val, maxval-1);
    }
    return val % maxval;
}

inline __device__ void get_weights_indices_jagged_chart(
    const int3 texture_info, const int3 texture_dim,
    const int channel, const float2 uv, const bool bilinear, const bool replicate_pad,
    float4& weights, int4& indices
) {
    int num_charts = texture_info.x;
    int c = texture_info.z;

    int h = texture_dim.x;
    int w = texture_dim.y;
    int si = texture_dim.z;
    float2 tcoord = {
        h * uv.x, w * uv.y
    };
    int2 iipixel = {
        (int)tcoord.x, (int)tcoord.y
    };
    int2 jjpixel = {
        fix_int_coord(iipixel.x + 1, h, replicate_pad),
        fix_int_coord(iipixel.y + 1, w, replicate_pad)
    };
    float2 weight = {
        tcoord.x - (float)iipixel.x, tcoord.y - (float)iipixel.y
    };
    iipixel.x = fix_int_coord(iipixel.x, h, replicate_pad);
    iipixel.y = fix_int_coord(iipixel.y, w, replicate_pad);
    const float wii = (1.f - weight.x) * (1.f - weight.y);
    const float wij = (1.f - weight.x) * (weight.y);
    const float wji = (weight.x) * (1.f - weight.y);
    const float wjj = (weight.x) * (weight.y);
    const int indii = (si + iipixel.x * w + iipixel.y) * c + channel;
    const int indij = (si + iipixel.x * w + jjpixel.y) * c + channel;
    const int indji = (si + jjpixel.x * w + iipixel.y) * c + channel;
    const int indjj = (si + jjpixel.x * w + jjpixel.y) * c + channel;
    if (bilinear) {
        weights = {wii, wij, wji, wjj};
    }
    else {
        if (wii >= wij && wii >= wji && wii >= wjj) {
            weights = {1.f, 0.f, 0.f, 0.f};
        }
        else if (wij >= wii && wij >= wji && wij >= wjj) {
            weights = {0.f, 1.f, 0.f, 0.f};
        }
        else if (wji >= wii && wji >= wij && wji >= wjj) {
            weights = {0.f, 0.f, 1.f, 0.f};
        }
        else {
            weights = {0.f, 0.f, 0.f, 1.f};
        }
    }

    indices = {indii, indij, indji, indjj};
}

inline __device__ void query_jagged_chart(
    const int3 texture_info, const int3 texture_dim, const int channel, const float2 uv,
    const float* __restrict__ texture, const bool bilinear, const bool replicate_pad,
    float4& weights, int4& indices, float4& corner_vals, float& val
) {
    get_weights_indices_jagged_chart(texture_info, texture_dim, channel, uv, bilinear, replicate_pad, weights, indices);
    const float wii = weights.x;
    const float wij = weights.y;
    const float wji = weights.z;
    const float wjj = weights.w;
    const int indii = indices.x;
    const int indij = indices.y;
    const int indji = indices.z;
    const int indjj = indices.w;
    const float tii = texture[indii];
    const float tij = texture[indij];
    const float tji = texture[indji];
    const float tjj = texture[indjj];
    corner_vals = {tii, tij, tji, tjj};
    val = wii * tii + wij * tij + wji * tji + wjj * tjj;
}

inline __device__ void edit_jagged_chart(
    const int3 texture_info, const int3 texture_dim, const int channel, const float2 uv,
    const float val, const bool replicate_pad, float* __restrict__ texture
) {
    float4 weights = {0.f, 0.f, 0.f, 0.f};
    int4 indices = {0, 0, 0, 0};
    get_weights_indices_jagged_chart(texture_info, texture_dim, channel, uv, true, replicate_pad, weights, indices);
    atomicAdd(texture + indices.x, weights.x * val);
    atomicAdd(texture + indices.y, weights.y * val);
    atomicAdd(texture + indices.z, weights.z * val);
    atomicAdd(texture + indices.w, weights.w * val);
}

inline __device__ void query_jagged_chart_vjp(
    const int3 texture_info, const int3 texture_dim, const int channel, const float2 uv,
    const float4 weights, const int4 indices, const float4 corner_vals, const bool bilinear, const bool replicate_pad,
    const float v_val, float2& v_uv, float* v_texture
) {
    atomicAdd(v_texture + indices.x, weights.x * v_val);
    atomicAdd(v_texture + indices.y, weights.y * v_val);
    atomicAdd(v_texture + indices.z, weights.z * v_val);
    atomicAdd(v_texture + indices.w, weights.w * v_val);

    if (bilinear) {
        int num_charts = texture_info.x;
        int c = texture_info.z;

        int h = texture_dim.x;
        int w = texture_dim.y;
        float2 tcoord = {
            h * uv.x, w * uv.y
        };
        int2 iipixel = {
            (int)tcoord.x, (int)tcoord.y
        };
        int2 jjpixel = {
            fix_int_coord(iipixel.x + 1, h, replicate_pad),
            fix_int_coord(iipixel.y + 1, w, replicate_pad)
        };
        float2 weight = {
            tcoord.x - (float)iipixel.x, tcoord.y - (float)iipixel.y
        };
        iipixel.x = fix_int_coord(iipixel.x, h, replicate_pad);
        iipixel.y = fix_int_coord(iipixel.y, w, replicate_pad);
        float2 v_weight = {0.f, 0.f};
        v_weight.x += v_val * (
            -(1.f - weight.y) * corner_vals.x
            -(weight.y) * corner_vals.y
            +(1.f - weight.y) * corner_vals.z
            +(weight.y) * corner_vals.w
        );
        v_weight.y += v_val * (
            -(1.f - weight.x) * corner_vals.x
            +(1.f - weight.x) * corner_vals.y
            -(weight.x) * corner_vals.z
            +(weight.x) * corner_vals.w
        );

        v_uv.x += h * v_weight.x;
        v_uv.y += w * v_weight.y;
    }
}

inline __device__ void compute_tt(const float3 origin, const float3 ray, const float3 mean, const float3 normal, float& t) {
    const float3 diff = {mean.x - origin.x, mean.y - origin.y, mean.z - origin.z};
    float t_denom = dot(normal, ray);
    float eps = 1e-6f;
    if (0.f <= t_denom && t_denom < eps) {
        t_denom = eps;
    }
    else if (-eps < t_denom && t_denom <= 0.f) {
        t_denom = -eps;
    }
    t = dot(normal, diff) / t_denom;
}

inline __device__ void compute_tt_vjp(
    const float3 origin, const float3 ray, const float3 mean, const float3 normal, const float t, const float v_tt,
    float3& v_mean, float3& v_normal
) {
    const float3 diff = {mean.x - origin.x, mean.y - origin.y, mean.z - origin.z};
    float t_denom = dot(normal, ray);
    float eps = 1e-6f;
    if (0.f <= t_denom && t_denom < eps) {
        t_denom = eps;
    }
    else if (-eps < t_denom && t_denom <= 0.f) {
        t_denom = -eps;
    }
    v_mean.x += normal.x * v_tt / t_denom;
    v_mean.y += normal.y * v_tt / t_denom;
    v_mean.z += normal.z * v_tt / t_denom;
    v_normal.x += (diff.x - ray.x * t) * v_tt / t_denom;
    v_normal.y += (diff.y - ray.y * t) * v_tt / t_denom;
    v_normal.z += (diff.z - ray.z * t) * v_tt / t_denom;
}

inline __device__ void get_ray(const float* __restrict__ c2w, float4 intrins, float px, float py, float3& origin, float3& ray) {
    float fx = intrins.x;
    float fy = intrins.y;
    float cx = intrins.z;
    float cy = intrins.w;
    origin = {c2w[3], c2w[7], c2w[11]};
    float u = (px - cx) / fx;
    float v = (py - cy) / fy;
    ray = transform_4x3(c2w, {u, v, 1});
    ray.x = ray.x - origin.x;
    ray.y = ray.y - origin.y;
    ray.z = ray.z - origin.z;
    float ray_norm = sqrtf(ray.x * ray.x + ray.y * ray.y + ray.z * ray.z);
    ray.x = ray.x / ray_norm;
    ray.y = ray.y / ray_norm;
    ray.z = ray.z / ray_norm;
}


inline __device__ void get_pixel_to_texel(const float* __restrict__ c2w, float4 intrins, float px, float py,
    const float3 mean, const float3 ax1, const float3 ax2, const float3 ax3, float& u, float& v) {
    float3 origin, ray;
    get_ray(c2w, intrins, px, py, origin, ray);
    float t = 0.f;
    compute_tt(origin, ray, mean, ax3, t);
    const float3 pos = {origin.x + t * ray.x, origin.y + t * ray.y, origin.z + t * ray.z};
    const float3 delta = {pos.x - mean.x, pos.y - mean.y, pos.z - mean.z};

    u = dot(delta, ax1);
    v = dot(delta, ax2);
}

inline __device__ void get_texel_area(const float* __restrict__ c2w, float4 intrins, float px, float py,
    const float3 mean, const float3 ax1, const float3 ax2, const float3 ax3, float& area) {
    // float3 origin, ray;
    // get_ray(c2w, intrins, px, py, origin, ray);
    // float sign = 1.f;
    // if (dot(ray, ax3) > 0.f) {
    //     sign = -1.f;
    // }

    float2 uv00, uv01, uv10, uv11;
    get_pixel_to_texel(c2w, intrins, px - 0.5f, py - 0.5f, mean, ax1, ax2, ax3, uv00.x, uv00.y);
    get_pixel_to_texel(c2w, intrins, px - 0.5f, py + 0.5f, mean, ax1, ax2, ax3, uv01.x, uv01.y);
    get_pixel_to_texel(c2w, intrins, px + 0.5f, py - 0.5f, mean, ax1, ax2, ax3, uv10.x, uv10.y);
    get_pixel_to_texel(c2w, intrins, px + 0.5f, py + 0.5f, mean, ax1, ax2, ax3, uv11.x, uv11.y);
    area = 0.5f * fabsf(
        (uv00.x * uv01.y - uv01.x * uv00.y) +
        (uv01.x * uv11.y - uv11.x * uv01.y) +
        (uv11.x * uv10.y - uv10.x * uv11.y) +
        (uv10.x * uv00.y - uv00.x * uv10.y)
    );
}

inline __device__ void local_outline(const float* __restrict__ c2w, float4 intrins, float px, float py,
    const float3 mean, const float3 ax1, const float3 ax2, const float3 ax3,
    const float scale1, const float scale2, int width, float sigma_thresh, float& dis) {
    // float3 origin, ray;
    // get_ray(c2w, intrins, px, py, origin, ray);
    // float sign = 1.f;
    // if (dot(ray, ax3) > 0.f) {
    //     sign = -1.f;
    // }
    float min_dis = 1000.0f;

    for (int dx = -width; dx <= width; dx++) {
        for (int dy = -width; dy <= width; dy++) {
            float2 uv;
            get_pixel_to_texel(c2w, intrins, px + (float)dx, py + (float)dy, mean, ax1, ax2, ax3, uv.x, uv.y);
            float as1 = uv.x / scale1;
            float as2 = uv.y / scale2;
            float sigma = 0.5f * (as1 * as1 + as2 * as2);
            if (sigma > sigma_thresh) {
                float cur_dis = sqrtf((float)(dx * dx + dy * dy));
                if (min_dis > cur_dis) {
                    min_dis = cur_dis;
                }
            }
        }
    }
    dis = min_dis;
}