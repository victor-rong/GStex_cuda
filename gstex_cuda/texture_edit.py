"""Python bindings for custom Cuda functions"""

from typing import Optional, Tuple

import torch
from jaxtyping import Float, Int
from torch import Tensor
from torch.autograd import Function

import gstex_cuda.cuda as _C
from .utils import bin_and_sort_gaussians, compute_cumulative_intersects
import gstex_cuda._torch_impl as _T

def texture_edit(
    texture_info: Tuple[Int, Int, Int],
    texture_dims: Int[Tensor, "td 3"],
    updated_img: Float[Tensor, "h w 3"],
    updated_alpha: Float[Tensor, "h w 1"],
    depth_lower: Float[Tensor, "h w"],
    depth_upper: Float[Tensor, "h w"],
    centers: Float[Tensor, "*batch 2"],
    extents: Float[Tensor, "*batch 2"],
    depths: Float[Tensor, "*batch 1"],
    num_tiles_hit: Int[Tensor, "*batch 1"],
    opacity: Float[Tensor, "*batch 1"],
    means: Float[Tensor, "*batch 3"],
    scales: Float[Tensor, "*batch 3"],
    glob_scale,
    quats: Float[Tensor, "*batch 4"],
    uv0: Float[Tensor, "*batch tch 2"],
    umap: Float[Tensor, "*batch tch 3"],
    vmap: Float[Tensor, "*batch tch 3"],
    viewmat: Float[Tensor, "4 4"],
    c2w: Float[Tensor, "4 4"],
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    img_height: int,
    img_width: int,
    block_width: int,
    settings: int,
    background: Optional[Float[Tensor, "channels"]] = None,
    use_torch_impl: bool = False,
) -> Tensor:
    """Rasterizes textured 2D gaussians

    Note:
        This function is differentiable w.r.t the Gaussian's parameters, the per-Gaussian UV mapping's
        parameters, and the texture values
    Args:
        texture_info (tuple): Information about number of textures and number of texture channels.
            Specifically, it should be (# of textures, 1, # of channels), although the first two
            entries are no longer used.
        texture_dims (Tensor): Information about the dimensions of each Gaussian's texture. Has shape
            (# of points, 3). The first channel is height, second is width, and final channel is
            starting index in the jagged tensor. Note that Gaussians can use the same texture, although
            this is not done in GStex.
        updated_img (Tensor): RGB canvas containing the desired edits on an empty canvas
        updated_alpha (Tensor): Alpha values of the same canvas. Areas that are not edited have 0 alpha
        depth_lower (Tensor): Per-pixel lower bound on the depth at which edits are made. Texels
            intersected at a depth lower than this are not processed.
        depth_upper (Tensor): Per-pixel upper bound on the depth at which edits are made. Texels
            intersected at a depth higher than this are not processed.
        centers (Tensor): Centers of screen-space AABBs containing Gaussians. Note that this is different
            from the projected Gaussian mean
        extents (Tensor): Extents of screen-space AABBs containing Gaussians
        depths (Tensor): Depths of Gaussian means. We use the same tile-based sorting by depth as 3DGS,
            hence there are still some multiview inconsistences from popping.
        num_tiles_hit (Tensor): number of tiles hit per gaussian
        opacity (Tensor): opacity associated with the gaussians.
        means (Tensor): Gaussian means
        scales (Tensor): Gaussian scales
        glob_scale (float): Global scale
        quats (Tensor): Quaternion describing Gaussian rotations
        uv0 (Tensor): Each Gaussian has a linear map from the Gaussian's plane to its texture, mapped to
            [0, 1] x [0, 1]. uv0 gives the texture coordinate where the Gaussian mean maps to.
            E.g. 0.5, 0.5 is chosen for GStex
        umap (Tensor): Parameters for the linear map described above. In particular,
            u = uv0.x + dot(umap, ax1)
        vmap (Tensor): Parameters for the linear map described above. In particular,
            v = uv0.y + dot(vmap, ax2)
        texture (Tensor): Jagged tensor containing all textures flattened and concatenated into a tensor
            of size (sum H*W, # of channels). Not actually used when editing.
        viewmat (Tensor): View matrix
        c2w (Tensor): Camera-to-world matrix
        fx: Camera intrinsics (focal length)
        fy: Camera intrinsics (focal length)
        cx: Camera intrinsics (principal point)
        cy: Camera intrinsics (principal point)
        img_height (int): height of the rendered image.
        img_width (int): width of the rendered image.
        block_width (int): MUST match whatever block width was used in the project_gaussians call.
            integer number of pixels between 2 and 16 inclusive
        settings (int): Various settings defined by its bits that are used by the CUDA implementation
        background (Tensor): Background color. Not actually used when editing.
        use_torch_impl (bool): Not used, only a CUDA implementation was written.

    Returns:
        updated_texture (Tensor): The texture which should be used in order to approximate updated_img
            and updated_alpha. Note that the first shape dimension is the same as texture, but it has
            5 channels: The first three are RGB, weighted by the intersection weights. The fourth is
            alpha, also weighted by the intersection weights. The last one is the total of the
            intersection weights. See also the paper supplementary.
    """
    assert block_width > 1 and block_width <= 16, "block_width must be between 2 and 16"

    func = _TextureEdit.apply

    return func(
        texture_info,
        texture_dims.contiguous(),
        updated_img.contiguous(),
        updated_alpha.contiguous(),
        depth_lower.contiguous(),
        depth_upper.contiguous(),
        centers.contiguous(),
        extents.contiguous(),
        depths.contiguous(),
        num_tiles_hit.contiguous(),
        opacity.contiguous(),
        means.contiguous(),
        scales.contiguous(),
        glob_scale,
        quats.contiguous(),
        uv0.contiguous(),
        umap.contiguous(),
        vmap.contiguous(),
        viewmat.contiguous(),
        c2w.contiguous(),
        fx,
        fy,
        cx,
        cy,
        img_height,
        img_width,
        block_width,
        settings,
        background.contiguous(),
    )

class _TextureEdit(Function):
    """Rasterizes 2D gaussians"""

    @staticmethod
    def apply(
        texture_info: Tuple[Int, Int, Int],
        texture_dims: Int[Tensor, "td 3"],
        updated_img: Float[Tensor, "h w 3"],
        updated_alpha: Float[Tensor, "h w 1"],
        depth_lower: Float[Tensor, "h w"],
        depth_upper: Float[Tensor, "h w"],
        centers: Float[Tensor, "*batch 2"],
        extents: Float[Tensor, "*batch 2"],
        depths: Float[Tensor, "*batch 1"],
        num_tiles_hit: Int[Tensor, "*batch 1"],
        opacity: Float[Tensor, "*batch 1"],
        means: Float[Tensor, "*batch 3"],
        scales: Float[Tensor, "*batch 3"],
        glob_scale,
        quats: Float[Tensor, "*batch 4"],
        uv0: Float[Tensor, "*batch tch 2"],
        umap: Float[Tensor, "*batch tch 3"],
        vmap: Float[Tensor, "*batch tch 3"],
        viewmat: Float[Tensor, "4 4"],
        c2w: Float[Tensor, "4 4"],
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        img_height: int,
        img_width: int,
        block_width: int,
        settings: int,
        background: Optional[Float[Tensor, "channels"]] = None,
    ) -> Tensor:
        num_points = centers.size(0)
        tile_bounds = (
            (img_width + block_width - 1) // block_width,
            (img_height + block_width - 1) // block_width,
            1,
        )
        block = (block_width, block_width, 1)
        img_size = (img_width, img_height, 1)
        texture_total_size = torch.sum(texture_dims[:,0] * texture_dims[:,1]).item()

        num_intersects, cum_tiles_hit = compute_cumulative_intersects(num_tiles_hit)
        
        if num_intersects < 1:
            assert False
        else:
            (
                isect_ids_unsorted,
                gaussian_ids_unsorted,
                isect_ids_sorted,
                gaussian_ids_sorted,
                tile_bins,
            ) = bin_and_sort_gaussians(
                num_points,
                num_intersects,
                centers,
                extents,
                depths,
                cum_tiles_hit,
                tile_bounds,
                block_width,
            )
            rasterize_fn = _C.texture_edit
            outputs = rasterize_fn(
                tile_bounds,
                block,
                img_size,
                texture_info,
                texture_total_size,
                texture_dims,
                updated_img,
                updated_alpha,
                depth_lower,
                depth_upper,
                gaussian_ids_sorted,
                tile_bins,
                opacity,
                means,
                scales,
                glob_scale,
                quats,
                uv0,
                umap,
                vmap,
                viewmat,
                c2w,
                fx,
                fy,
                cx,
                cy,
                settings,
                background,
            )
            updated_texture = outputs
        return updated_texture