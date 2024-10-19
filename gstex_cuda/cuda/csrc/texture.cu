#include "texture_helpers.cuh"
#include "texture.cuh"
#include <algorithm>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <iostream>
#include <cuda_fp16.h>

namespace cg = cooperative_groups;

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

    const bool compute_reg = true;
    const int depth_mode = 3;
    const bool use_blur = (settings & (1<<9));
    const bool use_ndc = (settings & (1<<10));
    const bool visualization_mode = (settings & (1<<15));
    const bool alpha_visualization_mode = (settings & (1<<16));
    const float alpha_bound = (float)((settings & (0b11111<<17))>>17) / 8.0f;
    const float outline_bound = (float)((settings & (0b1111<<26))>>26) / 4.0f;
    const bool accept_opacity_thresh = (settings & (1<<25));
    const bool white_outline = (settings & (1<<24));
    const bool bilinear_mode = !(settings & (1<<2));
    const bool use_replicate_pad = true;

    float px = (float)bj + 0.5;
    float py = (float)bi + 0.5;
    const int32_t pix_id = min(bi * img_size.x + bj, img_size.x * img_size.y - 1);

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
    const int block_size = block.size();
    int num_batches = (range.y - range.x + block_size - 1) / block_size;

    __shared__ int32_t id_batch[MAX_BLOCK_SIZE];
    __shared__ float3 means_batch[MAX_BLOCK_SIZE];
    __shared__ float opacities_batch[MAX_BLOCK_SIZE];
    __shared__ float3 scales_batch[MAX_BLOCK_SIZE];
    __shared__ float4 quats_batch[MAX_BLOCK_SIZE];
    __shared__ float3 rgbs_batch[MAX_BLOCK_SIZE];

    __shared__ int3 dims_batch[MAX_BLOCK_SIZE];
    __shared__ float2 uv0_batch[MAX_BLOCK_SIZE];
    __shared__ float3 umap_batch[MAX_BLOCK_SIZE];
    __shared__ float3 vmap_batch[MAX_BLOCK_SIZE];

    const int texture_channels = texture_info.z;

    float texture_buffer[64];
    float output_texture_buffer[64];

    for (int j = 0; j < texture_channels; j++) {
        texture_buffer[j] = 0.f;
        output_texture_buffer[j] = 0.f;
    }
    // current visibility left to render
    float T = 1.f;
    // index of most recent gaussian to write to this thread's pixel
    int cur_idx = 0;

    // collect and process batches of gaussians
    // each thread loads one gaussian at a time before rasterizing its
    // designated pixel
    int tr = block.thread_rank();
    float3 pix_out = {0.f, 0.f, 0.f};
    float3 pix_out_normal = {0.f, 0.f, 0.f};
    float pix_depth = 0.f;
    int pix_depth_index = -1;
    float pix_reg = 0.f;
    float max_vis = 0.f;

    float3 t_s = {0.f, 0.f, 0.f};

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
            rgbs_batch[tr] = colors[g_id];
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
                float alpha = min(0.99f, opac * (
                    non_blur_factor * __expf(-sigma) + blur_factor *__expf(-sigma_blur)
                ));
                float pixel_dis = 1000.0f;
                if (alpha_visualization_mode) {
                    alpha = 0.99f;
                    float sigma_thresh = 0.5f * alpha_bound * alpha_bound;
                    if (sigma > sigma_thresh) {
                        alpha = 0.0f;
                    }
                    if (accept_opacity_thresh && opac < 0.5f) {
                        alpha = 0.0f;
                    }
                    local_outline(c2w, intrins, px, py, mean, ax1, ax2, ax3, scale.x*glob_scale, scale.y*glob_scale, 3, sigma_thresh, pixel_dis);
                }
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
                    // int32_t g = id_batch[t];
                    const float vis = alpha * T;
                    const float3 c = rgbs_batch[i];
                    if (!alpha_visualization_mode || pixel_dis > outline_bound) {
                        pix_out.x = pix_out.x + c.x * vis;
                        pix_out.y = pix_out.y + c.y * vis;
                        pix_out.z = pix_out.z + c.z * vis;
                    }
                    else if (alpha_visualization_mode && white_outline) {
                        pix_out.x = pix_out.x + vis;
                        pix_out.y = pix_out.y + vis;
                        pix_out.z = pix_out.z + vis;
                    }
                    float3 h_ax3 = {ax3.x, ax3.y, ax3.z};
                    if (visualization_mode) {
                        if (ray.x * ax3.x + ray.y * ax3.y + ray.z * ax3.z > 0) {
                            h_ax3 = {-ax3.x, -ax3.y, -ax3.z};
                        }
                    }
                    pix_out_normal.x += vis * h_ax3.x;
                    pix_out_normal.y += vis * h_ax3.y;
                    pix_out_normal.z += vis * h_ax3.z;
                    const float t_view = t * view_depth;

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
                    for(int k = 0; k < texture_channels; k++) {
                        float val = 0.f;
                        float4 temp_weights;
                        int4 temp_indices;
                        float4 temp_values;
                        query_jagged_chart(
                            texture_info, dims, k, uv, texture, bilinear_mode, use_replicate_pad,
                            temp_weights, temp_indices, temp_values, val
                        );
                        if (!alpha_visualization_mode || pixel_dis > outline_bound) {
                            output_texture_buffer[k] += vis * val;
                        }
                        else if (alpha_visualization_mode && white_outline) {
                            output_texture_buffer[k] += vis;
                        }
                    }

                    if (depth_mode == 1) { //mean
                        pix_depth += t * vis;
                    }
                    else if (depth_mode == 2) { //mode
                        if (max_vis < vis) {
                            max_vis = vis;
                            pix_depth = t_view;
                        }
                    }
                    else if(depth_mode == 3) { //median
                        if (T > 0.5f) {
                            pix_depth = t_view;
                            pix_depth_index = batch_start + i;
                        }
                    }
                    if (compute_reg) {
                        if (use_ndc) {
                            const float t_near = 0.01f;
                            const float t_far = 1000.0f;
                            const float t_ndc = (t_far * t_view - t_far * t_near) / ((t_far - t_near) * t_view);
                            distortion(t_ndc, vis, t_s, pix_reg);
                        }
                        else {
                            distortion(t, vis, t_s, pix_reg);
                        }
                    }
                    T = next_T;
                    cur_idx = batch_start + i;
                }
            }
        }
    }

    if (inside) {
        // add background
        final_Ts[pix_id] = T; // transmittance at last gaussian in this pixel
        final_index[pix_id] =
            cur_idx; // index of in bin of last gaussian in this pixel
        float3 final_color;
        final_color.x = pix_out.x + T * background.x;
        final_color.y = pix_out.y + T * background.y;
        final_color.z = pix_out.z + T * background.z;
        depth_index[pix_id] = pix_depth_index;
        out_img[pix_id] = final_color;
        out_normal[pix_id] = pix_out_normal;
        out_depth[pix_id] = pix_depth;
        out_reg[pix_id] = pix_reg;
        out_reg_s[pix_id] = t_s;
        for (int j = 0; j < texture_channels; j++) {
            out_texture[texture_channels * pix_id + j] = output_texture_buffer[j];
        }
    }
}

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
) {
    auto block = cg::this_thread_block();
    int32_t tile_id =
        block.group_index().y * tile_bounds.x + block.group_index().x;
    unsigned bi =
        block.group_index().y * block.group_dim().y + block.thread_index().y;
    unsigned bj =
        block.group_index().x * block.group_dim().x + block.thread_index().x;

    const bool propagate_uv_grad = settings & (1<<8);
    const bool compute_reg = true;
    const int depth_mode = 3;
    const bool use_blur = (settings & (1<<9));
    const bool use_ndc = (settings & (1<<10));
    const bool bilinear_mode = !(settings & (1<<2));
    const bool use_replicate_pad = true;

    float px = (float)bj + 0.5;
    float py = (float)bi + 0.5;
    const int32_t pix_id = min(bi * img_size.x + bj, img_size.x * img_size.y - 1);

    // compute ray through pixel
    float3 origin, ray;
    get_ray(c2w, intrins, px, py, origin, ray);
    float view_depth = viewmat[8] * ray.x + viewmat[9] * ray.y + viewmat[10] * ray.z;

    // clamp this value to the last pixel
    // const int32_t pix_id = min(i * img_size.x + j, img_size.x * img_size.y - 1);

    // keep not rasterizing threads around for reading data
    const bool inside = (bi < img_size.y && bj < img_size.x);

    // this is the T AFTER the last gaussian in this pixel
    float T_final = final_Ts[pix_id];
    float T = T_final;
    float3 t_s_final = final_s[pix_id];
    int depth_final = depth_index[pix_id];
    // index of last gaussian to contribute to this pixel
    const int bin_final = inside? final_index[pix_id] : 0;

    // have all threads in tile process the same gaussians in batches
    // first collect gaussians between range.x and range.y in batches
    // which gaussians to look through in this tile
    const int2 range = tile_bins[tile_id];
    const int block_size = block.size();
    const int num_batches = (range.y - range.x + block_size - 1) / block_size;

    __shared__ int32_t id_batch[MAX_BLOCK_SIZE];
    __shared__ float3 means_batch[MAX_BLOCK_SIZE];
    __shared__ float opacities_batch[MAX_BLOCK_SIZE];
    __shared__ float3 scales_batch[MAX_BLOCK_SIZE];
    __shared__ float4 quats_batch[MAX_BLOCK_SIZE];
    __shared__ float3 rgbs_batch[MAX_BLOCK_SIZE];
    __shared__ int3 dims_batch[MAX_BLOCK_SIZE];
    __shared__ float2 uv0_batch[MAX_BLOCK_SIZE];
    __shared__ float3 umap_batch[MAX_BLOCK_SIZE];
    __shared__ float3 vmap_batch[MAX_BLOCK_SIZE];

    const int texture_channels = texture_info.z;

    float texture_buffer[64];
    float output_texture_buffer[64];
    for (int j = 0; j < texture_channels; j++) {
        texture_buffer[j] = 0.f;
        output_texture_buffer[j] = 0.f;
    }
    // df/d_out for this pixel
    const float3 v_out = v_output[pix_id];
    const float v_out_depth = v_output_depth[pix_id];
    const float v_out_reg = v_output_reg[pix_id];
    const float v_out_alpha = v_output_alpha[pix_id];
    const float3 v_out_normal = v_output_normal[pix_id];

    float v_output_texture_buffer[64];
    for (int j = 0; j < texture_channels; j++) {
        v_output_texture_buffer[j] = v_output_texture[texture_channels * pix_id + j];
    }
    float v_T_running = dot(background, v_out) - v_out_alpha;
    // collect and process batches of gaussians
    // each thread loads one gaussian at a time before rasterizing
    const int tr = block.thread_rank();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    const int warp_bin_final = cg::reduce(warp, bin_final, cg::greater<int>());
    for (int b = 0; b < num_batches; ++b) {
        // resync all threads before writing next batch of shared mem
        block.sync();

        // each thread fetch 1 gaussian from back to front
        // 0 index will be furthest back in batch
        // index of gaussian to load
        // batch end is the index of the last gaussian in the batch
        const int batch_end = range.y - 1 - block_size * b;
        int batch_size = min(block_size, batch_end + 1 - range.x);
        const int idx = batch_end - tr;
        if (idx >= range.x) {
            int32_t g_id = gaussian_ids_sorted[idx];
            id_batch[tr] = g_id;
            means_batch[tr] = means3d[g_id];
            opacities_batch[tr] = opacities[g_id];
            scales_batch[tr] = scales[g_id];
            quats_batch[tr] = quats[g_id];
            rgbs_batch[tr] = colors[g_id];
            dims_batch[tr] = texture_dims[g_id];
            uv0_batch[tr] = uv0s[g_id];
            umap_batch[tr] = umaps[g_id];
            vmap_batch[tr] = vmaps[g_id];
        }
        // wait for other threads to collect the gaussians in batch
        block.sync();
        // process gaussians in the current batch for this pixel
        // 0 index is the furthest back gaussian in the batch
        for (int i = max(0,batch_end - warp_bin_final); i < batch_size; ++i) {
            int valid = inside;
            if (batch_end - i > bin_final) {
                valid = 0;
            }
            const int g_id = id_batch[i];
            float alpha;
            float opac;
            float exp_sigma;
            float exp_sigma_blur;
            float3 scale;
            float3 mean;
            float4 quat;
            glm::mat3 R;
            float t;
            float t_view;
            const float t_near = 0.01f;
            const float t_far = 1000.0f;
            float t_ndc;
            float3 pos;
            float sigma_factor;
            float sigma_1;
            float sigma_2;
            float3 diff;
            float3 delta;
            float l1, l2, l3;
            float3 ax1, ax2, ax3;
            float3 p_view;
            float2 xy;
            float2 delta2d;

            const float blur_filtersize_sq = 2.0f;
            float non_blur_factor = 1.0f;
            float blur_factor = 0.0f;
            if(valid){
                scale = scales_batch[i];
                mean = means_batch[i];
                quat = quats_batch[i];
                opac = opacities_batch[i];
                R = quat_to_rotmat(quat);
                ax1 = {R[0][0], R[0][1], R[0][2]};
                ax2 = {R[1][0], R[1][1], R[1][2]};
                ax3 = {R[2][0], R[2][1], R[2][2]};

                compute_tt(origin, ray, mean, ax3, t);
                pos = {origin.x + t * ray.x, origin.y + t * ray.y, origin.z + t * ray.z};
                delta = {pos.x - mean.x, pos.y - mean.y, pos.z - mean.z};

                l1 = dot(delta, ax1);
                l2 = dot(delta, ax2);
                l3 = dot(delta, ax3);

                sigma_factor = 0.5f / (glob_scale * glob_scale);
                sigma_1 = l1 * l1 / (scale.x * scale.x);
                sigma_2 = l2 * l2 / (scale.y * scale.y);
                const float sigma = sigma_factor * (sigma_1 + sigma_2);

                p_view = transform_4x3(viewmat, mean);
                xy = project_pix({intrins.x, intrins.y}, p_view, {intrins.z, intrins.w});
                delta2d = {xy.x - px, xy.y - py};

                const float sigma_blur = 0.5f * blur_filtersize_sq * (delta2d.x * delta2d.x + delta2d.y * delta2d.y);
                if (use_blur) {
                    if (sigma_blur < sigma) {
                        blur_factor = 1.0f;
                        non_blur_factor = 0.0f;
                    }
                }
                exp_sigma = __expf(-sigma);
                exp_sigma_blur = __expf(-sigma_blur);
                alpha = min(0.99f, opac * (
                    non_blur_factor * exp_sigma + blur_factor * exp_sigma_blur
                ));
                if (t < 0.01f || t > 1000.0f || alpha < 1.f / 255.f) {
                    valid = 0;
                }
            }
            // if all threads are inactive in this warp, skip this loop
            if(!warp.any(valid)){
                continue;
            }
            float3 v_rgb_local = {0.f, 0.f, 0.f};
            float3 v_scale_local = {0.f, 0.f, 0.f};
            float3 v_mean_local = {0.f, 0.f, 0.f};
            float4 v_quat_local = {0.f, 0.f, 0.f, 0.f};
            float v_opacity_local = 0.f;
            float v_T_cur = 0.f;
            float3 v_normal_local = {0.f, 0.f, 0.f};
            float2 v_uv0_local = {0.f, 0.f};
            float3 v_umap_local = {0.f, 0.f, 0.f};
            float3 v_vmap_local = {0.f, 0.f, 0.f};
            //initialize everything to 0, only set if the lane is valid

            // if(valid){
            if(valid){
                // compute the current T for this gaussian
                float ra = 1.f / (1.f - alpha);
                T *= ra;
                // update v_rgb for this gaussian
                const float vis = alpha * T;
                v_rgb_local = {vis * v_out.x, vis * v_out.y, vis * v_out.z};

                v_normal_local = {vis * v_out_normal.x, vis * v_out_normal.y, vis * v_out_normal.z};
                const float3 rgb = rgbs_batch[i];

                float v_vis= dot(rgb, v_out);
                v_vis += dot(ax3, v_out_normal);


                float3 v_diff = {0.f, 0.f, 0.f};
                float v_alpha = 0.f;
                {
                    float2 uv = {0.f, 0.f};
                    float2 v_uv = {0.f, 0.f};
                    float v_weight = 0.f;

                    int3 dims = {0, 0, 0};
                    float2 uv0 = {0.f, 0.f};
                    float3 umap = {0.f, 0.f, 0.f};
                    float3 vmap = {0.f, 0.f, 0.f};

                    dims = dims_batch[i];
                    uv0 = uv0_batch[i];
                    umap = umap_batch[i];
                    vmap = vmap_batch[i];

                    get_uv(uv0, umap, vmap, delta, uv);
                    fix_coord(uv, use_replicate_pad);

                    for(int k = 0; k < texture_channels; k++) {
                        float val = 0.f;
                        float4 temp_weights = {0.f, 0.f, 0.f, 0.f};
                        int4 temp_indices = {0, 0, 0, 0};
                        float4 temp_values = {0.f, 0.f, 0.f, 0.f};
                        query_jagged_chart(
                            texture_info, dims, k, uv, texture, bilinear_mode, use_replicate_pad,
                            temp_weights, temp_indices, temp_values, val
                        );
                        float v_val = vis * v_output_texture_buffer[k];

                        float2 v_uv_detached = {0.f, 0.f};
                        query_jagged_chart_vjp(
                            texture_info, dims, k, uv,
                            temp_weights, temp_indices, temp_values, bilinear_mode, use_replicate_pad,
                            v_val, v_uv_detached, v_texture
                        );
                        if (propagate_uv_grad) {
                            v_uv.x += v_uv_detached.x;
                            v_uv.y += v_uv_detached.y;
                        }
                        v_vis += val * v_output_texture_buffer[k];
                    }
                    float3 v_diff_detached = {0.f, 0.f, 0.f};
                    get_uv_vjp(uv0, umap, vmap, delta, v_uv, v_uv0_local, v_umap_local, v_vmap_local, v_diff_detached);
                    if (propagate_uv_grad) {
                        v_diff.x += v_diff_detached.x;
                        v_diff.y += v_diff_detached.y;
                        v_diff.z += v_diff_detached.z;
                    }
                }

                float v_t_local = 0.f;
                float v_l1 = 0.f;
                float v_l2 = 0.f;
                glm::mat3 v_R_local;

                // contribution from this pixel
                v_alpha += T * v_vis - T * v_T_running;
                v_T_cur += alpha * v_vis + (1 - alpha) * v_T_running;

                float v_t_ndc_local = 0.f;
                float v_t_view_local = 0.f;
                if (compute_reg) {
                    t_view = t * view_depth;
                    t_ndc = (t_far * t_view - t_far * t_near) / ((t_far - t_near) * t_view);

                    float v_weight = 0.f;
                    if (use_ndc) {
                        distortion_vjp(t_ndc, vis, t_s_final, v_out_reg, v_t_ndc_local, v_weight);
                    }
                    else {
                        distortion_vjp(t, vis, t_s_final, v_out_reg, v_t_local, v_weight);
                    }

                    v_alpha += v_weight * T;
                    v_T_cur += v_weight * alpha;
                }
                v_T_running = v_T_cur;

                const float v_sigma = -non_blur_factor * opac * exp_sigma * v_alpha;
                v_scale_local = {-2 * sigma_factor * sigma_1 * v_sigma / scale.x, -2 * sigma_factor * sigma_2 * v_sigma / scale.y, 0.f};
                v_l1 += 2.f * sigma_factor * l1 * v_sigma / (scale.x * scale.x);
                v_l2 += 2.f * sigma_factor * l2 * v_sigma / (scale.y * scale.y);
                v_t_local += (v_l1 * dot(ray, ax1) + v_l2 * dot(ray, ax2));
                v_t_local += dot(ray, v_diff);
                if (depth_mode == 3 && batch_end - i == depth_final && depth_final != -1) {
                    v_t_view_local += v_out_depth;
                }
                v_t_view_local += (t_far * t_near) / ((t_far - t_near) * t_view * t_view) * v_t_ndc_local;
                v_t_local += view_depth * v_t_view_local;
                const float v_sigma_blur = -blur_factor * opac * exp_sigma_blur * v_alpha;

                float3 v_mean_extra = transform_4x3_rot_only_transposed(
                    viewmat,
                    project_pix_vjp(
                        {intrins.x, intrins.y},
                        p_view,
                        {blur_filtersize_sq * v_sigma_blur * (xy.x - px), blur_filtersize_sq * v_sigma_blur * (xy.y - py)}
                    )
                );
                float3 v_mean_tt = {0.f, 0.f, 0.f};
                float3 v_normal_tt = {0.f, 0.f, 0.f};
                compute_tt_vjp(origin, ray, mean, ax3, t, v_t_local, v_mean_tt, v_normal_tt);

                v_mean_local =
                    {-(ax1.x * v_l1 + ax2.x * v_l2) + v_mean_tt.x + v_mean_extra.x - v_diff.x,
                     -(ax1.y * v_l1 + ax2.y * v_l2) + v_mean_tt.y + v_mean_extra.y - v_diff.y,
                     -(ax1.z * v_l1 + ax2.z * v_l2) + v_mean_tt.z + v_mean_extra.z - v_diff.z};
                v_R_local = {
                    {delta.x * v_l1, delta.y * v_l1, delta.z * v_l1},
                    {delta.x * v_l2, delta.y * v_l2, delta.z * v_l2},
                    {v_normal_tt.x + v_normal_local.x, v_normal_tt.y + v_normal_local.y, v_normal_tt.z + v_normal_local.z}
                };
                v_quat_local = quat_to_rotmat_vjp(quat, v_R_local);
                v_opacity_local += (non_blur_factor * exp_sigma + blur_factor * exp_sigma_blur) * v_alpha;
            }
            warpSum3(v_rgb_local, warp);
            warpSum3(v_scale_local, warp);
            warpSum3(v_mean_local, warp);
            warpSum4(v_quat_local, warp);
            warpSum(v_opacity_local, warp);
            warpSum2(v_uv0_local, warp);
            warpSum3(v_umap_local, warp);
            warpSum3(v_vmap_local, warp);
            if (warp.thread_rank() == 0) {
                int32_t g = id_batch[i];

                float* v_rgb_ptr = (float*)(v_rgb);
                atomicAdd(v_rgb_ptr + 3*g + 0, v_rgb_local.x);
                atomicAdd(v_rgb_ptr + 3*g + 1, v_rgb_local.y);
                atomicAdd(v_rgb_ptr + 3*g + 2, v_rgb_local.z);
                
                float* v_scale_ptr = (float*)(v_scale);
                atomicAdd(v_scale_ptr + 3*g + 0, v_scale_local.x);
                atomicAdd(v_scale_ptr + 3*g + 1, v_scale_local.y);
                atomicAdd(v_scale_ptr + 3*g + 2, v_scale_local.z);
                
                float* v_mean_ptr = (float*)(v_mean);
                atomicAdd(v_mean_ptr + 3*g + 0, v_mean_local.x);
                atomicAdd(v_mean_ptr + 3*g + 1, v_mean_local.y);
                atomicAdd(v_mean_ptr + 3*g + 2, v_mean_local.z);
                
                float* v_quat_ptr = (float*)(v_quat);
                atomicAdd(v_quat_ptr + 4*g + 0, v_quat_local.x);
                atomicAdd(v_quat_ptr + 4*g + 1, v_quat_local.y);
                atomicAdd(v_quat_ptr + 4*g + 2, v_quat_local.z);
                atomicAdd(v_quat_ptr + 4*g + 3, v_quat_local.w);

                atomicAdd(v_opacity + g, v_opacity_local);

                float* v_uv0_ptr = (float*)(v_uv0);
                float* v_umap_ptr = (float*)(v_umap);
                float* v_vmap_ptr = (float*)(v_vmap);

                atomicAdd(v_umap_ptr + 3*g + 0, v_umap_local.x);
                atomicAdd(v_umap_ptr + 3*g + 1, v_umap_local.y);
                atomicAdd(v_umap_ptr + 3*g + 2, v_umap_local.z);

                atomicAdd(v_vmap_ptr + 3*g + 0, v_vmap_local.x);
                atomicAdd(v_vmap_ptr + 3*g + 1, v_vmap_local.y);
                atomicAdd(v_vmap_ptr + 3*g + 2, v_vmap_local.z);

                atomicAdd(v_uv0_ptr + 2*g + 0, v_uv0_local.x);
                atomicAdd(v_uv0_ptr + 2*g + 1, v_uv0_local.y);
            }
        }
    }
}

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
) {
    CHECK_INPUT(texture_dims);
    CHECK_INPUT(gaussian_ids_sorted);
    CHECK_INPUT(tile_bins);
    CHECK_INPUT(colors);
    CHECK_INPUT(opacities);
    CHECK_INPUT(means);
    CHECK_INPUT(scales);
    CHECK_INPUT(quats);
    CHECK_INPUT(uv0);
    CHECK_INPUT(umap);
    CHECK_INPUT(vmap);
    CHECK_INPUT(texture);
    CHECK_INPUT(viewmat);
    CHECK_INPUT(c2w);
    CHECK_INPUT(background);
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

    const int channels = colors.size(1);
    const int img_width = img_size_dim3.x;
    const int img_height = img_size_dim3.y;

    torch::Tensor out_img = torch::zeros(
        {img_height, img_width, channels}, means.options().dtype(torch::kFloat32)
    );
    torch::Tensor final_Ts = torch::zeros(
        {img_height, img_width}, means.options().dtype(torch::kFloat32)
    );
    torch::Tensor final_idx = torch::zeros(
        {img_height, img_width}, means.options().dtype(torch::kInt32)
    );
    torch::Tensor out_depth = torch::zeros(
        {img_height, img_width}, means.options().dtype(torch::kFloat32)
    );
    torch::Tensor depth_idx = torch::zeros(
        {img_height, img_width}, means.options().dtype(torch::kInt32)
    );
    torch::Tensor out_reg = torch::zeros(
        {img_height, img_width}, means.options().dtype(torch::kFloat32)
    );
    torch::Tensor out_reg_s = torch::zeros(
        {img_height, img_width, 3}, means.options().dtype(torch::kFloat32)
    );
    torch::Tensor out_texture = torch::zeros(
        {img_height, img_width, texture_channels}, means.options().dtype(torch::kFloat32)
    );
    torch::Tensor out_normal = torch::zeros(
        {img_height, img_width, 3}, means.options().dtype(torch::kFloat32)
    );
    texture_forward<<<tile_bounds_dim3, block_dim3>>>(
        tile_bounds_dim3,
        img_size_dim3,
        texture_info_int3,
        (int3 *)texture_dims.contiguous().data_ptr<int>(),
        gaussian_ids_sorted.contiguous().data_ptr<int32_t>(),
        (int2 *)tile_bins.contiguous().data_ptr<int>(),
        (float3 *)colors.contiguous().data_ptr<float>(),
        opacities.contiguous().data_ptr<float>(),
        (float3 *)means.contiguous().data_ptr<float>(),
        (float3 *)scales.contiguous().data_ptr<float>(),
        glob_scale,
        (float4 *)quats.contiguous().data_ptr<float>(),
        (float2 *)uv0.contiguous().data_ptr<float>(),
        (float3 *)umap.contiguous().data_ptr<float>(),
        (float3 *)vmap.contiguous().data_ptr<float>(),
        texture.contiguous().data_ptr<float>(),
        viewmat.contiguous().data_ptr<float>(),
        c2w.contiguous().data_ptr<float>(),
        intrins,
        settings,
        *(float3 *)background.contiguous().data_ptr<float>(),
        final_Ts.contiguous().data_ptr<float>(),
        final_idx.contiguous().data_ptr<int>(),
        depth_idx.contiguous().data_ptr<int>(),
        (float3 *)out_reg_s.contiguous().data_ptr<float>(),
        (float3 *)out_img.contiguous().data_ptr<float>(),
        out_depth.contiguous().data_ptr<float>(),
        out_reg.contiguous().data_ptr<float>(),
        out_texture.contiguous().data_ptr<float>(),
        (float3 *)out_normal.contiguous().data_ptr<float>()
    );

    return std::make_tuple(
        out_img, out_depth, out_reg, out_texture, out_normal,
        final_Ts, final_idx, depth_idx, out_reg_s
    );
}

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
    ) {
    CHECK_INPUT(texture_dims);
    CHECK_INPUT(gaussian_ids_sorted);
    CHECK_INPUT(tile_bins);
    CHECK_INPUT(colors);
    CHECK_INPUT(opacities);
    CHECK_INPUT(means);
    CHECK_INPUT(scales);
    CHECK_INPUT(quats);
    CHECK_INPUT(uv0);
    CHECK_INPUT(umap);
    CHECK_INPUT(vmap);
    CHECK_INPUT(texture);
    CHECK_INPUT(viewmat);
    CHECK_INPUT(c2w);
    CHECK_INPUT(background);
    CHECK_INPUT(final_Ts);
    CHECK_INPUT(final_idx);
    CHECK_INPUT(depth_idx);
    CHECK_INPUT(final_s);
    CHECK_INPUT(v_output);
    CHECK_INPUT(v_output_depth);
    CHECK_INPUT(v_output_reg);
    CHECK_INPUT(v_output_alpha);
    CHECK_INPUT(v_output_texture);
    CHECK_INPUT(v_output_normal);
    float4 intrins = {fx, fy, cx, cy};

    const int num_points = means.size(0);
    const dim3 tile_bounds = {
        (img_width + block_width - 1) / block_width,
        (img_height + block_width - 1) / block_width,
        1
    };
    const dim3 block(block_width, block_width, 1);
    const dim3 img_size = {img_width, img_height, 1};
    const int channels = colors.size(1);
    int3 texture_info_int3;
    texture_info_int3.x = std::get<0>(texture_info);
    texture_info_int3.y = std::get<1>(texture_info);
    texture_info_int3.z = std::get<2>(texture_info);
    int num_charts = texture_info_int3.x;
    int num_probs = texture_info_int3.y;
    int texture_channels = texture_info_int3.z;

    torch::Tensor v_xys = torch::zeros({num_points, 2}, means.options());
    torch::Tensor v_conics = torch::zeros({num_points, 3}, means.options());
    torch::Tensor v_colors = torch::zeros({num_points, channels}, means.options());
    torch::Tensor v_opacity = torch::zeros({num_points, 1}, means.options());
    torch::Tensor v_means = torch::zeros({num_points, 3}, means.options());
    torch::Tensor v_scales = torch::zeros({num_points, 3}, means.options());
    torch::Tensor v_quats = torch::zeros({num_points, 4}, means.options());
    torch::Tensor v_prob = torch::zeros({num_points, num_probs}, means.options());
    torch::Tensor v_uv0 = torch::zeros({num_points, num_probs, 2}, means.options());
    torch::Tensor v_umap = torch::zeros({num_points, num_probs, 3}, means.options());
    torch::Tensor v_vmap = torch::zeros({num_points, num_probs, 3}, means.options());
    torch::Tensor v_texture = torch::zeros({texture.size(0), texture_channels}, means.options());

    texture_backward<<<tile_bounds, block>>>(
        tile_bounds,
        img_size,
        texture_info_int3,
        (int3 *)texture_dims.contiguous().data_ptr<int>(),
        gaussian_ids_sorted.contiguous().data_ptr<int32_t>(),
        (int2 *)tile_bins.contiguous().data_ptr<int>(),
        (float3 *)colors.contiguous().data_ptr<float>(),
        opacities.contiguous().data_ptr<float>(),
        (float3 *)means.contiguous().data_ptr<float>(),
        (float3 *)scales.contiguous().data_ptr<float>(),
        glob_scale,
        (float4 *)quats.contiguous().data_ptr<float>(),
        (float2 *)uv0.contiguous().data_ptr<float>(),
        (float3 *)umap.contiguous().data_ptr<float>(),
        (float3 *)vmap.contiguous().data_ptr<float>(),
        texture.contiguous().data_ptr<float>(),
        viewmat.contiguous().data_ptr<float>(),
        c2w.contiguous().data_ptr<float>(),
        intrins,
        settings,
        *(float3 *)background.contiguous().data_ptr<float>(),
        final_Ts.contiguous().data_ptr<float>(),
        final_idx.contiguous().data_ptr<int>(),
        depth_idx.contiguous().data_ptr<int>(),
        (float3 *)final_s.contiguous().data_ptr<float>(),
        (float3 *)v_output.contiguous().data_ptr<float>(),
        v_output_depth.contiguous().data_ptr<float>(),
        v_output_reg.contiguous().data_ptr<float>(),
        v_output_alpha.contiguous().data_ptr<float>(),
        v_output_texture.contiguous().data_ptr<float>(),
        (float3 *)v_output_normal.contiguous().data_ptr<float>(),
        (float3 *)v_colors.contiguous().data_ptr<float>(),
        v_opacity.contiguous().data_ptr<float>(),
        (float3 *)v_means.contiguous().data_ptr<float>(),
        (float3 *)v_scales.contiguous().data_ptr<float>(),
        (float4 *)v_quats.contiguous().data_ptr<float>(),
        (float2 *)v_uv0.contiguous().data_ptr<float>(),
        (float3 *)v_umap.contiguous().data_ptr<float>(),
        (float3 *)v_vmap.contiguous().data_ptr<float>(),
        v_texture.contiguous().data_ptr<float>()
    );

    return std::make_tuple(v_colors, v_opacity, v_means, v_scales, v_quats, v_uv0, v_umap, v_vmap, v_texture);
}