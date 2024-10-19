#include "texture_helpers.cuh"
#include "texture_sample.cuh"
#include <algorithm>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <iostream>
#include <cuda_fp16.h>

namespace cg = cooperative_groups;

__global__ void texture_sample_forward(
    const int num_queries,
    const int3 texture_info, //{num_charts, num_prob, num_channels}
    const int3* texture_dims, //{h, w, start_index}
    const float2* __restrict__ uvs,
    const float* __restrict__ texture,
    float* __restrict__ output
) {
    unsigned idx = cg::this_grid().thread_rank();
    if (idx >= num_queries)
        return;
    const int num_channels = texture_info.z;
    const int3 dims = texture_dims[idx];
    const float2 uv = uvs[idx];
    for (int i = 0; i < num_channels; i++) {
        float val = 0.f;
        float4 temp_weights;
        int4 temp_indices;
        float4 temp_values;
        query_jagged_chart(
            texture_info, dims, i, uv, texture, true, true,
            temp_weights, temp_indices, temp_values, val
        );
        output[idx * num_channels + i] = val;
    }
}

__global__ void texture_sample_backward(
    const int num_queries,
    const int3 texture_info, //{num_charts, num_prob, num_channels}
    const int3* texture_dims, //{h, w, start_index}
    const float2* __restrict__ uvs,
    const float* __restrict__ texture,
    const float* __restrict__ v_output,
    float* __restrict__ v_texture
) {
    unsigned idx = cg::this_grid().thread_rank();
    if (idx >= num_queries)
        return;
    const int num_channels = texture_info.z;
    const int3 dims = texture_dims[idx];
    const float2 uv = uvs[idx];
    for (int i = 0; i < num_channels; i++) {
        float val = 0.f;
        float4 temp_weights;
        int4 temp_indices;
        float4 temp_values;
        float v_val = v_texture[num_channels * idx + i];
        float2 v_uv;
        query_jagged_chart(
            texture_info, dims, i, uv, texture, true, true,
            temp_weights, temp_indices, temp_values, val
        );
        query_jagged_chart_vjp(
            texture_info, dims, i, uv,
            temp_weights, temp_indices, temp_values, true, true,
            v_val, v_uv, v_texture
        );
    }
}

torch::Tensor texture_sample_forward_tensor(
    const std::tuple<int, int, int> texture_info,
    const torch::Tensor &texture_dims,
    const torch::Tensor &uvs,
    const torch::Tensor &texture
) {
    CHECK_INPUT(texture_dims);
    CHECK_INPUT(uvs);
    CHECK_INPUT(texture);

    int3 texture_info_int3;
    texture_info_int3.x = std::get<0>(texture_info);
    texture_info_int3.y = std::get<1>(texture_info);
    texture_info_int3.z = std::get<2>(texture_info);

    const int num_queries = uvs.size(0);
    const int value_channels = texture_info_int3.z;

    torch::Tensor output = torch::zeros(
        {num_queries, value_channels}, texture.options().dtype(torch::kFloat32)
    );
    texture_sample_forward<<<
        (num_queries + N_THREADS - 1) / N_THREADS,
        N_THREADS>>>(
        num_queries,
        texture_info_int3,
        (int3 *)texture_dims.contiguous().data_ptr<int>(),
        (float2 *)uvs.contiguous().data_ptr<float>(),
        texture.contiguous().data_ptr<float>(),
        output.contiguous().data_ptr<float>()
    );

    return output;
}

torch::Tensor texture_sample_backward_tensor(
    const std::tuple<int, int, int> texture_info,
    const torch::Tensor &texture_dims,
    const torch::Tensor &uvs,
    const torch::Tensor &texture,
    const torch::Tensor &v_output
) {
    CHECK_INPUT(texture_dims);
    CHECK_INPUT(uvs);
    CHECK_INPUT(texture);

    int3 texture_info_int3;
    texture_info_int3.x = std::get<0>(texture_info);
    texture_info_int3.y = std::get<1>(texture_info);
    texture_info_int3.z = std::get<2>(texture_info);

    const int num_queries = uvs.size(0);
    const int value_channels = texture_info_int3.z;

    torch::Tensor v_texture = torch::zeros({texture.size(0), value_channels}, texture.options());

    texture_sample_backward<<<
        (num_queries + N_THREADS - 1) / N_THREADS,
        N_THREADS>>>(
        num_queries,
        texture_info_int3,
        (int3 *)texture_dims.contiguous().data_ptr<int>(),
        (float2 *)uvs.contiguous().data_ptr<float>(),
        texture.contiguous().data_ptr<float>(),
        v_output.contiguous().data_ptr<float>(),
        v_texture.contiguous().data_ptr<float>()
    );

    return v_texture;
}
