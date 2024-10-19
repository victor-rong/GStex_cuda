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
);

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
);