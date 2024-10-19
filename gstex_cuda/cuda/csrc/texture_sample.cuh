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

__global__ void texture_sample_forward(
    const int num_queries,
    const int3 texture_info, //{num_charts, num_prob, num_channels}
    const int3* texture_dims, //{h, w, start_index}
    const float2* __restrict__ uvs,
    const float* __restrict__ texture,
    float* __restrict__ output
);

__global__ void texture_sample_backward(
    const int num_queries,
    const int3 texture_info, //{num_charts, num_prob, num_channels}
    const int3* texture_dims, //{h, w, start_index}
    const float2* __restrict__ uvs,
    const float* __restrict__ texture,
    const float* __restrict__ v_output,
    float* __restrict__ v_texture
);

torch::Tensor texture_sample_forward_tensor(
    const std::tuple<int, int, int> texture_info,
    const torch::Tensor &texture_dims,
    const torch::Tensor &uvs,
    const torch::Tensor &texture
);

torch::Tensor texture_sample_backward_tensor(
    const std::tuple<int, int, int> texture_info,
    const torch::Tensor &texture_dims,
    const torch::Tensor &uvs,
    const torch::Tensor &texture,
    const torch::Tensor &v_output
);