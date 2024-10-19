"""Python bindings for custom Cuda functions"""

from typing import Optional

import torch
from jaxtyping import Float, Int
from torch import Tensor
from torch.autograd import Function

import gstex_cuda.cuda as _C
import gstex_cuda._torch_impl as _T

def texture_sample(
    texture_info, texture_dims, texture, uvs, use_torch_impl: bool = False,
) -> Tensor:
    """Differentiable sampling from a jagged tensor. Performs arbitrary number of queries.

    Args:
        texture_info (tuple): Information about number of textures and number of texture channels.
            Specifically, it should be (# of textures, 1, # of channels), although the first two
            entries are no longer used.
        texture_dims (Tensor): Information about which texture each query is querying into. Has shape
            (# of queries, 3). The first channel is height, second is width, and final channel is
            starting index in the jagged tensor. Note that multiple queries can query into the same
            texture.
        texture (Tensor): Jagged tensor containing all textures flattened and concatenated into a tensor
            of size (sum H*W, # of channels)
        uvs (Tensor): Texture coordinates of each query in [0, 1] x [0, 1]. These are clamped to
            [0, 1] x [0, 1]. The gradient of the CUDA version is currently incorrect for clamped
            instances, but by default the gradient is detached anyways.
        use_torch_impl (bool): Whether to use the torch implementation

    Returns:
        out_value (Tensor): Tensor of shape (# of queries, # of channels). Results from performing the
            queries into the jagged tensor.
    """
    if use_torch_impl:
        func = _TextureSampleTorch.apply
    else:
        func = _TextureSample.apply

    return func(
        texture_info,
        texture_dims.contiguous(),
        texture.contiguous(),
        uvs.contiguous(),
    )

class _TextureSample(Function):
    @staticmethod
    def forward(
        ctx, texture_info, texture_dims, texture, uvs
    ) -> Tensor:
        rasterize_fn = _C.texture_sample_forward
        out_value = rasterize_fn(
            texture_info,
            texture_dims,
            uvs,
            texture,
        )
        
        ctx.texture_info = texture_info
        ctx.save_for_backward(
            texture_dims,
            texture,
            uvs,
        )
        return out_value

    @staticmethod
    def backward(ctx, v_out_value):
        
        return (
            None,  # texture_info
            None,  # texture_dims
            None,  # texture
            None,  # uvs
        )

class _TextureSampleTorch(Function):
    def apply(texture_info, texture_dims, texture, uvs) -> Tensor:
        return _T.sample_texture(texture_dims, texture, uvs)