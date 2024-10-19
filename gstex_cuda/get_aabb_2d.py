"""Python bindings for custom Cuda functions"""

from typing import Optional

import torch
from jaxtyping import Float, Int
from torch import Tensor
from torch.autograd import Function
import numpy as np

import gstex_cuda.cuda as _C
from .utils import bin_and_sort_gaussians, compute_cumulative_intersects
import gstex_cuda._torch_impl as _T

def merge_aabbs(centers1, extents1, centers2, extents2):
    lower = torch.minimum(centers1 - extents1, centers2 - extents2)
    upper = torch.maximum(centers1 + extents1, centers2 + extents2)
    centers = 0.5 * (lower + upper)
    extents = 0.5 * (upper - lower)
    return centers, extents

def project_points(points, viewmat, intrins, clip=False):
    view_points = points @ viewmat.T[:3,:3] + viewmat.T[3:,:3]
    if clip:
        t_min = 0.01
        t_max = 1000.0
        with torch.no_grad():
            view_points[:,-1] = torch.clamp(view_points[:,-1], min=t_min, max=t_max)

    pix = _T.project_pix(intrins[:2], view_points, intrins[2:])
    depths = view_points[:,-1]
    return pix, depths

def get_aabb_2d_torch(means, scales, glob_scale, quats, viewmat, intrins):
    ell = 3.0 * glob_scale
    Rs = _T.normalized_quat_to_rotmat(quats)

    corners = torch.stack([
        means + ell * scales[:,None,0] * Rs[:,:,0] + ell * scales[:,None,1] * Rs[:,:,1],
        means + ell * scales[:,None,0] * Rs[:,:,0] - ell * scales[:,None,1] * Rs[:,:,1],
        means - ell * scales[:,None,0] * Rs[:,:,0] + ell * scales[:,None,1] * Rs[:,:,1],
        means - ell * scales[:,None,0] * Rs[:,:,0] - ell * scales[:,None,1] * Rs[:,:,1],
    ], dim=1)
    old_shape = corners.shape

    corners = corners.reshape(-1, 3)
    projected_corners = project_points(corners, viewmat, intrins, clip=True)[0].reshape(old_shape[0], old_shape[1], 2)
    pix_max_corner = torch.max(projected_corners, dim=1)[0]
    pix_min_corner = torch.min(projected_corners, dim=1)[0]

    centers = 0.5 * (pix_max_corner + pix_min_corner)
    extents = 0.5 * (pix_max_corner - pix_min_corner)

    return centers, extents

def get_aabb_2d(means, scales, glob_scale, quats, viewmat, intrins):
    fx, fy, cx, cy = intrins
    return _C.get_aabb_2d(
        means.contiguous(),
        scales.contiguous(),
        glob_scale,
        quats.contiguous(),
        viewmat.contiguous(),
        fx,
        fy,
        cx,
        cy,
    )

def get_num_tiles_hit_2d(centers, extents, img_height, img_width, block_width):
    tile_bounds = (
        (img_width + block_width - 1) // block_width,
        (img_height + block_width - 1) // block_width,
    )
    top_left = torch.floor((centers - extents) / block_width).to(torch.int32)
    bottom_right = torch.floor((centers + extents) / block_width + 1).to(torch.int32)
    tile_min = torch.stack(
        [
            torch.clamp(top_left[..., 0], 0, tile_bounds[0]),
            torch.clamp(top_left[..., 1], 0, tile_bounds[1]),
        ],
        -1,
    )
    tile_max = torch.stack(
        [
            torch.clamp(bottom_right[..., 0], 0, tile_bounds[0]),
            torch.clamp(bottom_right[..., 1], 0, tile_bounds[1]),
        ],
        -1,
    )
    num_tiles_hit = (tile_max - tile_min)[:,0] * (tile_max - tile_min)[:,1]
    return num_tiles_hit