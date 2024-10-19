"""Python bindings for custom Cuda functions"""

from typing import Optional, Tuple

import torch
from jaxtyping import Float, Int
from torch import Tensor
from torch.autograd import Function

import gstex_cuda.cuda as _C
from .utils import bin_and_sort_gaussians, compute_cumulative_intersects
import gstex_cuda._torch_impl as _T

def texture_gaussians(
    texture_info: Tuple[Int, Int, Int],
    texture_dims: Int[Tensor, "td 3"],
    centers: Float[Tensor, "*batch 2"],
    extents: Float[Tensor, "*batch 2"],
    depths: Float[Tensor, "*batch 1"],
    num_tiles_hit: Int[Tensor, "*batch 1"],
    colors: Float[Tensor, "*batch channels"],
    opacity: Float[Tensor, "*batch 1"],
    means: Float[Tensor, "*batch 3"],
    scales: Float[Tensor, "*batch 3"],
    glob_scale,
    quats: Float[Tensor, "*batch 4"],
    uv0: Float[Tensor, "*batch tch 2"],
    umap: Float[Tensor, "*batch tch 3"],
    vmap: Float[Tensor, "*batch tch 3"],
    texture: Float[Tensor, "tch tc"],
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
        centers (Tensor): Centers of screen-space AABBs containing Gaussians. Note that this is different
            from the projected Gaussian mean
        extents (Tensor): Extents of screen-space AABBs containing Gaussians
        depths (Tensor): Depths of Gaussian means. We use the same tile-based sorting by depth as 3DGS,
            hence there are still some multiview inconsistences from popping.
        num_tiles_hit (Tensor): number of tiles hit per gaussian
        colors (Tensor): N-dimensional features associated with the gaussians.
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
            of size (sum H*W, # of channels)
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
        background (Tensor): background color
        use_torch_impl (bool): Whether to use the torch implementation

    Returns:
        out_img (Tensor): Rendered output image using values from colors (invariant across Gaussian)
        out_depth (Tensor): Median depth as defined in 2DGS
        out_reg (Tensor): L2 distortion loss of NDC depths as defined in 2DGS
        out_alpha (Tensor): Alpha channel of the rendered output image
        out_texture (Tensor): Rendered output image using values from texture (varies across Gaussian)
        out_normal (Tensor): Rendered normals
    """
    assert block_width > 1 and block_width <= 16, "block_width must be between 2 and 16"
    if colors.dtype == torch.uint8:
        # make sure colors are float [0,1]
        colors = colors.float() / 255

    if background is not None:
        assert (
            background.shape[0] == colors.shape[-1]
        ), f"incorrect shape of background color tensor, expected shape {colors.shape[-1]}"
    else:
        background = torch.ones(
            colors.shape[-1], dtype=torch.float32, device=colors.device
        )

    if colors.ndimension() != 2:
        raise ValueError("colors must have dimensions (N, D)")

    if use_torch_impl:
        func = _TextureGaussiansTorch.apply
    else:
        func = _TextureGaussians.apply

    return func(
        texture_info,
        texture_dims.contiguous(),
        centers.contiguous(),
        extents.contiguous(),
        depths.contiguous(),
        num_tiles_hit.contiguous(),
        colors.contiguous(),
        opacity.contiguous(),
        means.contiguous(),
        scales.contiguous(),
        glob_scale,
        quats.contiguous(),
        uv0.contiguous(),
        umap.contiguous(),
        vmap.contiguous(),
        texture.contiguous(),
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

class _TextureGaussians(Function):
    """Rasterizes textured 2D gaussians"""

    @staticmethod
    def forward(
        ctx,
        texture_info: Tuple[Int, Int, Int],
        texture_dims: Int[Tensor, "td 3"],
        centers: Float[Tensor, "*batch 2"],
        extents: Float[Tensor, "*batch 2"],
        depths: Float[Tensor, "*batch 1"],
        num_tiles_hit: Int[Tensor, "*batch 1"],
        colors: Float[Tensor, "*batch channels"],
        opacity: Float[Tensor, "*batch 1"],
        means: Float[Tensor, "*batch 3"],
        scales: Float[Tensor, "*batch 3"],
        glob_scale,
        quats: Float[Tensor, "*batch 4"],
        uv0: Float[Tensor, "*batch tch 2"],
        umap: Float[Tensor, "*batch tch 3"],
        vmap: Float[Tensor, "*batch tch 3"],
        texture: Float[Tensor, "tch tc"],
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

        num_intersects, cum_tiles_hit = compute_cumulative_intersects(num_tiles_hit)
        
        if num_intersects < 1:
            out_img = (
                torch.ones(img_height, img_width, colors.shape[-1], device=centers.device)
                * background
            )
            gaussian_ids_sorted = torch.zeros(0, 1, device=centers.device)
            tile_bins = torch.zeros(0, 2, device=centers.device)
            final_Ts = torch.zeros(img_height, img_width, device=centers.device)
            final_idx = torch.zeros(img_height, img_width, device=centers.device)
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
            rasterize_fn = _C.texture_forward
            outputs = rasterize_fn(
                tile_bounds,
                block,
                img_size,
                texture_info,
                texture_dims,
                gaussian_ids_sorted,
                tile_bins,
                colors,
                opacity,
                means,
                scales,
                glob_scale,
                quats,
                uv0,
                umap,
                vmap,
                texture,
                viewmat,
                c2w,
                fx,
                fy,
                cx,
                cy,
                settings,
                background,
            )
            out_img, out_depth, out_reg, out_texture, out_normal = outputs[:5]
            final_Ts, final_idx, depth_idx, out_reg_s = outputs[5:9]

        ctx.img_width = img_width
        ctx.img_height = img_height
        ctx.texture_info = texture_info
        ctx.fx = fx
        ctx.fy = fy
        ctx.cx = cx
        ctx.cy = cy
        ctx.glob_scale = glob_scale
        ctx.num_intersects = num_intersects
        ctx.block_width = block_width
        ctx.settings = settings
        ctx.save_for_backward(
            texture_dims,
            gaussian_ids_sorted,
            tile_bins,
            colors,
            opacity,
            means,
            scales,
            quats,
            uv0,
            umap,
            vmap,
            texture,
            viewmat,
            c2w,
            background,
            final_Ts,
            final_idx,
            depth_idx,
            out_reg_s,
        )
        out_alpha = 1 - final_Ts
        return (
            out_img, out_depth, out_reg, out_alpha, out_texture, out_normal
        )

    @staticmethod
    def backward(
        ctx, v_out_img, v_out_depth, v_out_reg, v_out_alpha, v_out_texture, v_out_normal
    ):
        img_height = ctx.img_height
        img_width = ctx.img_width
        num_intersects = ctx.num_intersects
        texture_info = ctx.texture_info
        fx = ctx.fx
        fy = ctx.fy
        cx = ctx.cx
        cy = ctx.cy
        glob_scale = ctx.glob_scale
        block_width = ctx.block_width
        settings = ctx.settings
        if v_out_alpha is None:
            v_out_alpha = torch.zeros_like(v_out_img[..., 0])

        (
            texture_dims,
            gaussian_ids_sorted,
            tile_bins,
            colors,
            opacity,
            means,
            scales,
            quats,
            uv0,
            umap,
            vmap,
            texture,
            viewmat,
            c2w,
            background,
            final_Ts,
            final_idx,
            depth_idx,
            out_reg_s,
        ) = ctx.saved_tensors

        if num_intersects < 1:
            v_colors = torch.zeros_like(colors)
            v_opacity = torch.zeros_like(opacity)
            v_means = torch.zeros_like(means)
            v_scales = torch.zeros_like(scales)
            v_quats = torch.zeros_like(quats)
            v_uv0 = torch.zeros_like(uv0)
            v_umap = torch.zeros_like(umap)
            v_vmap = torch.zeros_like(vmap)
            v_texture = torch.zeros_like(texture)
        else:
            rasterize_fn = _C.texture_backward
            v_colors, v_opacity, v_means, v_scales, v_quats, v_uv0, v_umap, v_vmap, v_texture = rasterize_fn(
                img_height,
                img_width,
                block_width,
                texture_info,
                texture_dims,
                gaussian_ids_sorted,
                tile_bins,
                colors,
                opacity,
                means,
                scales,
                glob_scale,
                quats,
                uv0,
                umap,
                vmap,
                texture,
                viewmat,
                c2w,
                fx,
                fy,
                cx,
                cy,
                settings,
                background,
                final_Ts,
                final_idx,
                depth_idx,
                out_reg_s,
                v_out_img,
                v_out_depth,
                v_out_reg,
                v_out_alpha,
                v_out_texture,
                v_out_normal,
            )
        return (
            None,  # texture_info
            None,  # texture_dims
            None,  # centers
            None,  # extents
            None,  # depths
            None,  # num_tiles_hit
            v_colors,  # colors
            v_opacity,  # opacity
            v_means,    # means
            v_scales,   # scales
            None,  # glob_scale
            v_quats,    # quats
            v_uv0,  # uv0
            v_umap,  # umap
            v_vmap,  # vmap
            v_texture,  # texture
            None,  # view_mat
            None,  # c2w
            None,  # fx
            None,  # fy
            None,  # cx
            None,  # cy
            None,  # img_height
            None,  # img_width
            None,  # block_width
            None,  # settings
            None,  # background
        )


class _TextureGaussiansTorch(Function):
    """Rasterizes textured 2D gaussians"""

    def apply(
        texture_info: Tuple[Int, Int, Int],
        texture_dims: Int[Tensor, "td 3"],
        centers: Float[Tensor, "*batch 2"],
        extents: Float[Tensor, "*batch 2"],
        depths: Float[Tensor, "*batch 1"],
        num_tiles_hit: Int[Tensor, "*batch 1"],
        colors: Float[Tensor, "*batch channels"],
        opacity: Float[Tensor, "*batch 1"],
        means: Float[Tensor, "*batch 3"],
        scales: Float[Tensor, "*batch 3"],
        glob_scale,
        quats: Float[Tensor, "*batch 4"],
        uv0: Float[Tensor, "*batch tch 2"],
        umap: Float[Tensor, "*batch tch 3"],
        vmap: Float[Tensor, "*batch tch 3"],
        texture: Float[Tensor, "tch tc"],
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

        num_intersects, cum_tiles_hit = compute_cumulative_intersects(num_tiles_hit)

        if num_intersects < 1:
            out_img = (
                torch.ones(img_height, img_width, colors.shape[-1], device=centers.device)
                * background
            )
            gaussian_ids_sorted = torch.zeros(0, 1, device=centers.device)
            tile_bins = torch.zeros(0, 2, device=centers.device)
            final_Ts = torch.zeros(img_height, img_width, device=centers.device)
            final_idx = torch.zeros(img_height, img_width, device=centers.device)
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
            rasterize_fn = _T.texture_forward
            (
                out_img, out_depth, out_reg, out_texture, out_normal, final_Ts, final_idx
            ) = rasterize_fn(
                tile_bounds,
                block,
                img_size,
                texture_info,
                texture_dims,
                gaussian_ids_sorted,
                tile_bins,
                colors,
                opacity,
                means,
                scales,
                glob_scale,
                quats,
                uv0,
                umap,
                vmap,
                texture,
                viewmat,
                c2w,
                fx,
                fy,
                cx,
                cy,
                settings,
                background,
            )

        out_alpha = 1 - final_Ts
        return (
            out_img, out_depth, out_reg, out_alpha, out_texture, out_normal
        )
