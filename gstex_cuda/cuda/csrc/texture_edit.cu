#include "texture_helpers.cuh"
#include "texture_edit.cuh"
#include <algorithm>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <iostream>
#include <cuda_fp16.h>

namespace cg = cooperative_groups;

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
    float* __restrict__ updated_texture // c = 5, (r, g, b, a, total)
) {
    // each thread draws one pixel, but also timeshares caching gaussians in a
    // shared tile

    auto block = cg::this_thread_block();
    int32_t tile_id =
        block.group_index().y * tile_bounds.x + block.group_index().x;
    unsigned bi =
        block.group_index().y * block.group_dim().y + block.thread_index().y;
    unsigned bj =
        block.group_index().x * block.group_dim().x + block.thread_index().x;

    const bool use_blur = (settings & (1<<0));
    const bool use_ndc = (settings & (1<<1));
    const bool use_replicate_pad = true;

    float px = (float)bj + 0.5;
    float py = (float)bi + 0.5;
    const int32_t pix_id = min(bi * img_size.x + bj, img_size.x * img_size.y - 1);

    const float3 rgb_update = updated_img[pix_id];
    const float mask_update = updated_alpha[pix_id];

    // compute ray through pixel
    float3 origin, ray;
    get_ray(c2w, intrins, px, py, origin, ray);
    float view_depth = viewmat[8] * ray.x + viewmat[9] * ray.y + viewmat[10] * ray.z;

    // return if out of bounds
    // keep not rasterizing threads around for reading data
    bool inside = (bi < img_size.y && bj < img_size.x);
    bool done = !inside;

    // have all threads in tile process the same gaussians in batches
    // first collect gaussians between range.x and range.y in batches
    // which gaussians to look through in this tile
    int2 range = tile_bins[tile_id];
    const float pix_depth_lower = depth_lower[pix_id];
    const float pix_depth_upper = depth_upper[pix_id];
    const int block_size = block.size();
    int num_batches = (range.y - range.x + block_size - 1) / block_size;

    __shared__ int32_t id_batch[MAX_BLOCK_SIZE];
    __shared__ float3 means_batch[MAX_BLOCK_SIZE];
    __shared__ float opacities_batch[MAX_BLOCK_SIZE];
    __shared__ float3 scales_batch[MAX_BLOCK_SIZE];
    __shared__ float4 quats_batch[MAX_BLOCK_SIZE];

    __shared__ int3 dims_batch[MAX_BLOCK_SIZE];
    __shared__ float2 uv0_batch[MAX_BLOCK_SIZE];
    __shared__ float3 umap_batch[MAX_BLOCK_SIZE];
    __shared__ float3 vmap_batch[MAX_BLOCK_SIZE];

    const int num_charts = texture_info.x;
    const int num_probs = texture_info.y;
    const int texture_channels = texture_info.z;

    // current visibility left to render
    float T = 1.f;
    // index of most recent gaussian to write to this thread's pixel
    int cur_idx = 0;

    // collect and process batches of gaussians
    // each thread loads one gaussian at a time before rasterizing its
    // designated pixel
    int tr = block.thread_rank();

    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    for (int b = 0; b < num_batches; ++b) {
        // resync all threads before beginning next batch
        // end early if entire tile is done
        if (__syncthreads_count(done) >= block_size) {
            break;
        }

        // each thread fetch 1 gaussian from front to back
        // index of gaussian to load
        int batch_start = range.x + block_size * b;
        int idx = batch_start + tr;
        if (idx < range.y) {
            int32_t g_id = gaussian_ids_sorted[idx];
            id_batch[tr] = g_id;
            means_batch[tr] = means3d[g_id];
            opacities_batch[tr] = opacities[g_id];
            scales_batch[tr] = scales[g_id];
            quats_batch[tr] = quats[g_id];
            dims_batch[tr] = texture_dims[g_id];
            uv0_batch[tr] = uv0s[g_id];
            umap_batch[tr] = umaps[g_id];
            vmap_batch[tr] = vmaps[g_id];
        }

        // wait for other threads to collect the gaussians in batch
        block.sync();
        // process gaussians in the current batch for this pixel
        int batch_size = min(block_size, range.y - batch_start);
        for (int i = 0; (i < batch_size); ++i) {
            float gaussian_weight = 0.f;
            bool skip = false;
            if (!done) {
                const int g_id = id_batch[i];
                const float3 scale = scales_batch[i];
                const float3 mean = means_batch[i];
                const float4 quat = quats_batch[i];
                const float opac = opacities_batch[i];
                glm::mat3 R = quat_to_rotmat(quat);
                const float3 ax1 = {R[0][0], R[0][1], R[0][2]};
                const float3 ax2 = {R[1][0], R[1][1], R[1][2]};
                const float3 ax3 = {R[2][0], R[2][1], R[2][2]};
                float t;
                compute_tt(origin, ray, mean, ax3, t);
                const float3 pos = {origin.x + t * ray.x, origin.y + t * ray.y, origin.z + t * ray.z};
                const float3 delta = {pos.x - mean.x, pos.y - mean.y, pos.z - mean.z};

                const float l1 = dot(delta, ax1);
                const float l2 = dot(delta, ax2);
                const float l3 = dot(delta, ax3);

                const float iscale1 = 1/(scale.x * glob_scale);
                const float iscale2 = 1/(scale.y * glob_scale);
                const float sigma = 0.5f * (iscale1 * iscale1 * l1 * l1 + iscale2 * iscale2 * l2 * l2);
                const float exp_sigma = expf(-sigma);

                float3 p_view = transform_4x3(viewmat, mean);
                float2 xy = project_pix({intrins.x, intrins.y}, p_view, {intrins.z, intrins.w});
                const float blur_filtersize_sq = 2.0f;
                float2 delta2d = {xy.x - px, xy.y - py};
                const float sigma_blur = 0.5f * blur_filtersize_sq * (delta2d.x * delta2d.x + delta2d.y * delta2d.y);

                float non_blur_factor = 1.0f;
                float blur_factor = 0.0f;

                if (use_blur) {
                    if (sigma_blur < sigma) {
                        blur_factor = 1.0f;
                        non_blur_factor = 0.0f;
                    }
                }
                const float alpha = min(0.99f, opac * (
                    non_blur_factor * __expf(-sigma) + blur_factor *__expf(-sigma_blur)
                ));
                if (t < 0.01f || t > 1000.0f || alpha < 1.f / 255.f) {
                    skip = true;
                }
                const float next_T = T * (1.f - alpha);
                if (next_T <= 1e-4f) { // this pixel is done
                    // we want to render the last gaussian that contributes and note
                    // that here idx > range.x so we don't underflow
                    done = true;
                }
                if (!skip && !done) {
                    const float vis = alpha * T;
                    const float t_view = t * view_depth;
                    bool in_depth_range = (t_view >= pix_depth_lower) && (t_view <= pix_depth_upper);
                    if (in_depth_range) {
                        int3 dims = {0, 0, 0};
                        float2 uv = {0.f, 0.f};
                        float2 uv0 = {0.f, 0.f};
                        float3 umap = {0.f, 0.f, 0.f};
                        float3 vmap = {0.f, 0.f, 0.f};

                        dims = dims_batch[i];
                        uv0 = uv0_batch[i];
                        umap = umap_batch[i];
                        vmap = vmap_batch[i];
                        get_uv(uv0, umap, vmap, delta, uv);
                        fix_coord(uv, use_replicate_pad);

                        for(int k = 0; k < 5; k++) {
                            float val = 0.f;
                            if (k == 0) {
                                val = rgb_update.x * mask_update;
                            }
                            else if (k == 1) {
                                val = rgb_update.y * mask_update;
                            }
                            else if (k == 2) {
                                val = rgb_update.z * mask_update;
                            }
                            else if (k == 3) {
                                val = mask_update;
                            }
                            else {
                                val = 1.f;
                            }
                            float4 temp_weights;
                            int4 temp_indices;
                            float4 temp_values;
                            edit_jagged_chart(
                                texture_info, dims, k, uv, val, use_replicate_pad, updated_texture
                            );
                        }
                    }
                    T = next_T;
                    cur_idx = batch_start + i;
                }
            }
        }
    }
}

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
) {
    CHECK_INPUT(texture_dims);
    CHECK_INPUT(gaussian_ids_sorted);
    CHECK_INPUT(tile_bins);
    CHECK_INPUT(opacities);
    CHECK_INPUT(means);
    CHECK_INPUT(scales);
    CHECK_INPUT(quats);
    CHECK_INPUT(uv0);
    CHECK_INPUT(umap);
    CHECK_INPUT(vmap);
    CHECK_INPUT(viewmat);
    CHECK_INPUT(c2w);
    CHECK_INPUT(background);
    CHECK_INPUT(depth_lower);
    CHECK_INPUT(depth_upper);
    CHECK_INPUT(updated_img);
    CHECK_INPUT(updated_alpha);
    float4 intrins = {fx, fy, cx, cy};

    dim3 tile_bounds_dim3;
    tile_bounds_dim3.x = std::get<0>(tile_bounds);
    tile_bounds_dim3.y = std::get<1>(tile_bounds);
    tile_bounds_dim3.z = std::get<2>(tile_bounds);

    dim3 block_dim3;
    block_dim3.x = std::get<0>(block);
    block_dim3.y = std::get<1>(block);
    block_dim3.z = std::get<2>(block);

    dim3 img_size_dim3;
    img_size_dim3.x = std::get<0>(img_size);
    img_size_dim3.y = std::get<1>(img_size);
    img_size_dim3.z = std::get<2>(img_size);

    int3 texture_info_int3;
    texture_info_int3.x = std::get<0>(texture_info);
    texture_info_int3.y = std::get<1>(texture_info);
    texture_info_int3.z = std::get<2>(texture_info);
    int num_charts = texture_info_int3.x;
    int num_probs = texture_info_int3.y;
    int texture_channels = texture_info_int3.z;

    const int channels = 3;
    const int img_width = img_size_dim3.x;
    const int img_height = img_size_dim3.y;

    torch::Tensor updated_texture = torch::zeros(
        {texture_total_size, texture_channels}, means.options().dtype(torch::kFloat32)
    );
    
    texture_edit<<<tile_bounds_dim3, block_dim3>>>(
        tile_bounds_dim3,
        img_size_dim3,
        texture_info_int3,
        (int3 *)texture_dims.contiguous().data_ptr<int>(),
        (float3 *)updated_img.contiguous().data_ptr<float>(),
        updated_alpha.contiguous().data_ptr<float>(),
        depth_lower.contiguous().data_ptr<float>(),
        depth_upper.contiguous().data_ptr<float>(),
        gaussian_ids_sorted.contiguous().data_ptr<int32_t>(),
        (int2 *)tile_bins.contiguous().data_ptr<int>(),
        opacities.contiguous().data_ptr<float>(),
        (float3 *)means.contiguous().data_ptr<float>(),
        (float3 *)scales.contiguous().data_ptr<float>(),
        glob_scale,
        (float4 *)quats.contiguous().data_ptr<float>(),
        (float2 *)uv0.contiguous().data_ptr<float>(),
        (float3 *)umap.contiguous().data_ptr<float>(),
        (float3 *)vmap.contiguous().data_ptr<float>(),
        viewmat.contiguous().data_ptr<float>(),
        c2w.contiguous().data_ptr<float>(),
        intrins,
        settings,
        *(float3 *)background.contiguous().data_ptr<float>(),
        updated_texture.contiguous().data_ptr<float>()
    );

    return updated_texture;
}