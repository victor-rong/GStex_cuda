"""Pure PyTorch implementations of various functions"""

import struct

import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor
from typing import Tuple


def compute_sh_color(
    viewdirs: Float[Tensor, "*batch 3"], sh_coeffs: Float[Tensor, "*batch D C"]
):
    """
    :param viewdirs (*, C)
    :param sh_coeffs (*, D, C) sh coefficients for each color channel
    return colors (*, C)
    """
    *dims, dim_sh, C = sh_coeffs.shape
    bases = eval_sh_bases(dim_sh, viewdirs)  # (*, dim_sh)
    return (bases[..., None] * sh_coeffs).sum(dim=-2)


"""
Taken from https://github.com/sxyu/svox2
"""

SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199
SH_C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396,
]
SH_C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435,
]
SH_C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]

MAX_SH_BASIS = 10


def eval_sh_bases(basis_dim: int, dirs: torch.Tensor):
    """
    Evaluate spherical harmonics bases at unit directions,
    without taking linear combination.
    At each point, the final result may the be
    obtained through simple multiplication.

    :param basis_dim: int SH basis dim. Currently, 1-25 square numbers supported
    :param dirs: torch.Tensor (..., 3) unit directions

    :return: torch.Tensor (..., basis_dim)
    """
    result = torch.empty(
        (*dirs.shape[:-1], basis_dim), dtype=dirs.dtype, device=dirs.device
    )
    result[..., 0] = SH_C0
    if basis_dim > 1:
        x, y, z = dirs.unbind(-1)
        result[..., 1] = -SH_C1 * y
        result[..., 2] = SH_C1 * z
        result[..., 3] = -SH_C1 * x
        if basis_dim > 4:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result[..., 4] = SH_C2[0] * xy
            result[..., 5] = SH_C2[1] * yz
            result[..., 6] = SH_C2[2] * (2.0 * zz - xx - yy)
            result[..., 7] = SH_C2[3] * xz
            result[..., 8] = SH_C2[4] * (xx - yy)

            if basis_dim > 9:
                result[..., 9] = SH_C3[0] * y * (3 * xx - yy)
                result[..., 10] = SH_C3[1] * xy * z
                result[..., 11] = SH_C3[2] * y * (4 * zz - xx - yy)
                result[..., 12] = SH_C3[3] * z * (2 * zz - 3 * xx - 3 * yy)
                result[..., 13] = SH_C3[4] * x * (4 * zz - xx - yy)
                result[..., 14] = SH_C3[5] * z * (xx - yy)
                result[..., 15] = SH_C3[6] * x * (xx - 3 * yy)

                if basis_dim > 16:
                    result[..., 16] = SH_C4[0] * xy * (xx - yy)
                    result[..., 17] = SH_C4[1] * yz * (3 * xx - yy)
                    result[..., 18] = SH_C4[2] * xy * (7 * zz - 1)
                    result[..., 19] = SH_C4[3] * yz * (7 * zz - 3)
                    result[..., 20] = SH_C4[4] * (zz * (35 * zz - 30) + 3)
                    result[..., 21] = SH_C4[5] * xz * (7 * zz - 3)
                    result[..., 22] = SH_C4[6] * (xx - yy) * (7 * zz - 1)
                    result[..., 23] = SH_C4[7] * xz * (xx - 3 * yy)
                    result[..., 24] = SH_C4[8] * (
                        xx * (xx - 3 * yy) - yy * (3 * xx - yy)
                    )
    return result


def normalized_quat_to_rotmat(quat: Tensor) -> Tensor:
    assert quat.shape[-1] == 4, quat.shape
    w, x, y, z = torch.unbind(quat, dim=-1)
    mat = torch.stack(
        [
            1 - 2 * (y**2 + z**2),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x**2 + z**2),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x**2 + y**2),
        ],
        dim=-1,
    )
    return mat.reshape(quat.shape[:-1] + (3, 3))


def quat_to_rotmat(quat: Tensor) -> Tensor:
    assert quat.shape[-1] == 4, quat.shape
    return normalized_quat_to_rotmat(F.normalize(quat, dim=-1))

def project_pix(fxfy, p_view, center, eps=1e-6):
    fx, fy = fxfy
    cx, cy = center

    rw = 1.0 / (p_view[..., 2] + 1e-6)
    p_proj = (p_view[..., 0] * rw, p_view[..., 1] * rw)
    u, v = (p_proj[0] * fx + cx, p_proj[1] * fy + cy)
    return torch.stack([u, v], dim=-1)

def sample_texture(texture_dim, texture, uvs):
    replicate_pad = True
    if replicate_pad:
        uu = torch.clamp(uvs[:,0], min=0, max=1)
        vv = torch.clamp(uvs[:,1], min=0, max=1)
    else:
        uu = torch.remainder(uvs[:,0], 1)
        vv = torch.remainder(uvs[:,1], 1)

    h = texture_dim[:,0]
    w = texture_dim[:,1]
    si = texture_dim[:,2]
    coords = torch.stack(
        [
            h * uu,
            w * vv,
        ],
        dim=-1
    )
    iicoords = torch.floor(coords)
    weight = coords - iicoords
    iicoords = iicoords.to(dtype=torch.int32)
    jjcoords = iicoords + 1
    if replicate_pad:
        iicoords[:,0] = torch.clamp(iicoords[:,0], max=h-1)
        iicoords[:,1] = torch.clamp(iicoords[:,1], max=w-1)
        jjcoords[:,0] = torch.clamp(jjcoords[:,0], max=h-1)
        jjcoords[:,1] = torch.clamp(jjcoords[:,1], max=w-1)
    else:
        iicoords[:,0] = iicoords[:,0] % h
        iicoords[:,1] = iicoords[:,1] % w
        jjcoords[:,0] = jjcoords[:,0] % h
        jjcoords[:,1] = jjcoords[:,1] % w

    idx00 = si + iicoords[:,0] * w + iicoords[:,1]
    idx01 = si + iicoords[:,0] * w + jjcoords[:,1]
    idx10 = si + jjcoords[:,0] * w + iicoords[:,1]
    idx11 = si + jjcoords[:,0] * w + jjcoords[:,1]

    local_sample = (
        ((1 - weight[:,0]) * (1 - weight[:,1]))[:,None] * texture[idx00,:] +
        ((1 - weight[:,0]) * (weight[:,1]))[:,None] * texture[idx01,:] +
        ((weight[:,0]) * (1 - weight[:,1]))[:,None] * texture[idx10,:] +
        ((weight[:,0]) * (weight[:,1]))[:,None] * texture[idx11,:]
    )
    return local_sample
    
def texture_forward(
    tile_bounds,
    block,
    img_size,
    texture_info,
    texture_dims,
    gaussian_ids_sorted,
    tile_bins,
    colors,
    opacities,
    means3d,
    scales,
    glob_scale,
    quats,
    uv0s,
    umaps,
    vmaps,
    texture,
    viewmat,
    c2w,
    fx,
    fy,
    cx,
    cy,
    settings,
    background,
):
    depth_mode = 3
    compute_reg = True
    channels = colors.shape[1]
    device = colors.device
    out_img = torch.zeros(
        (img_size[1], img_size[0], channels), dtype=torch.float32, device=device
    )
    out_depth = torch.zeros(
        (img_size[1], img_size[0]), dtype=torch.float32, device=device
    )
    out_reg = torch.zeros(
        (img_size[1], img_size[0]), dtype=torch.float32, device=device
    )
    out_texture = torch.zeros(
        (img_size[1], img_size[0], texture_info[-1]), dtype=torch.float32, device=device
    )
    out_normal = torch.zeros(
        (img_size[1], img_size[0], 3), dtype=torch.float32, device=device
    )
    final_Ts = torch.ones(
        (img_size[1], img_size[0]), dtype=torch.float32, device=device
    )
    final_idx = torch.zeros(
        (img_size[1], img_size[0]), dtype=torch.int32, device=device
    )
    origin = torch.zeros((4, 1), device=device)
    origin[3] = 1
    origin = (c2w @ origin).squeeze(-1)
    origin = origin[:3]
    Rs = normalized_quat_to_rotmat(quats)  # (..., 3, 3)

    for ti in range(tile_bounds[1]):
        for tj in range(tile_bounds[0]):
            ti_l = ti * block[1]
            ti_r = min(ti_l + block[1], img_size[1])
            tj_l = tj * block[0]
            tj_r = min(tj_l + block[0], img_size[0])
            image_jis = torch.stack(
                torch.meshgrid(torch.arange(tj_l, tj_r), torch.arange(ti_l, ti_r)),
                dim=-1
            ).to(device=device)
            image_jis = image_jis.reshape(-1, 2)

            ndc_x = (image_jis[:,0] - cx + 0.5) / fx
            ndc_y = (image_jis[:,1] - cy + 0.5) / fy
            rays = torch.stack((ndc_x, ndc_y, torch.ones_like(ndc_x), torch.zeros_like(ndc_x)), dim=-1) @ c2w.T
            rays = rays / (torch.sqrt(torch.sum(rays**2, dim=-1, keepdim=True)))
            view_depth = viewmat[2,0] * rays[:,0] + viewmat[2,1] * rays[:,1] + viewmat[2,2] * rays[:,2]
            rays = rays[:,:3]

            tile_id = ti * tile_bounds[0] + tj
            tile_bin_start = tile_bins[tile_id, 0]
            tile_bin_end = tile_bins[tile_id, 1]
            final_idx_tile = torch.zeros((image_jis.shape[0],), device=device, dtype=torch.int32)
            s0 = torch.zeros((image_jis.shape[0],), device=device, dtype=torch.float32)
            s1 = torch.zeros((image_jis.shape[0],), device=device, dtype=torch.float32)
            s2 = torch.zeros((image_jis.shape[0],), device=device, dtype=torch.float32)

            idx = tile_bin_end
            l = tile_bin_start
            r = tile_bin_end
            if l == r:
                continue
            gaussian_ids = gaussian_ids_sorted[l:r]

            R = Rs[gaussian_ids,:,:]
            scale = glob_scale * scales[gaussian_ids,:]
            mean = means3d[gaussian_ids,:]

            t_denom = torch.sum(R[:,None,:,2] * rays[None,:,:], dim=-1)
            eps = 1e-6
            t_denom[(0 <= t_denom) & (t_denom < eps)] = eps
            t_denom[(-eps <= t_denom) & (t_denom <= 0)] = -eps
            t = torch.sum(R[:,:,2] * (mean - origin), dim=-1)[:,None] / t_denom
            pos = origin[None,None,:] + t[:,:,None] * rays[None,:,:]
            delta = pos[:,:,:] - mean[:,None,:]
            ax1 = R[:,None,:,0]
            ax2 = R[:,None,:,1]
            ax3 = R[:,None,:,2]
            l1 = torch.sum(ax1 * delta[:,:,:], dim=-1)
            l2 = torch.sum(ax2 * delta[:,:,:], dim=-1)
            l3 = torch.sum(ax3 * delta[:,:,:], dim=-1)
            sigma = 0.5 * (l1 * l1 / (scale[:,None,0])**2 + l2 * l2 / (scale[:,None,1])**2)

            opac = opacities[gaussian_ids]

            pxy = image_jis.to(torch.float32) + 0.5
            # We don't do the 2D anti-aliasing in this pytorch implementation, but it's done in the CUDA implementation
            
            alpha = torch.clamp(opac * torch.exp(-sigma), max=0.999)

            invalid_mask = ((alpha < 1/255) | (t < 0.01) | (t > 1000.0))
            alpha[invalid_mask] = 0.0

            color = colors[gaussian_ids]
            uv0 = uv0s[gaussian_ids]
            umap = umaps[gaussian_ids]
            vmap = vmaps[gaussian_ids]
            tdim = texture_dims[gaussian_ids]
            T_true = torch.cat([torch.ones_like(alpha[:1,:]), 1-alpha[:,:]], dim=0).cumprod(dim=0)
            T = torch.zeros_like(T_true[:-1,:])
            T_mask = T_true[1:,:] <= 1e-4
            T[T_mask] = 0.0
            T[~T_mask] = T_true[:-1,:][~T_mask]
            vis = alpha * T

            acc_alpha = vis.sum(dim=0)

            texture_samples = torch.zeros(
                (t.shape[0], t.shape[1], texture.shape[-1]), device=texture.device
            )
            if not (settings & (1<<8)):
                delta = delta.detach()
            eps = 0.0
            uu = torch.clamp(uv0[:,0,None,0] + torch.sum(delta * umap, dim=-1), min=eps, max=1 - eps)
            vv = torch.clamp(uv0[:,0,None,1] + torch.sum(delta * vmap, dim=-1), min=eps, max=1 - eps)

            if not (settings & (1<<8)):
                uu = uu.detach()
                vv = vv.detach()
            tdi = tdim[:,None,:].repeat(1, uu.shape[1], 1).reshape(-1, 3)
            reshaped_uvs = torch.stack([uu, vv], dim=-1).reshape(-1, 2)
            local_sample = sample_texture(tdi, texture, reshaped_uvs)
            local_sample = local_sample.reshape(*uu.shape[:2], -1)
            texture_samples += local_sample

            out_img[image_jis[:,0],image_jis[:,1],:] = (vis[:,:,None] * color[:,None,:]).sum(dim=0) + (1-acc_alpha[:,None]) * background[None,:]
            out_normal[image_jis[:,0],image_jis[:,1],:] = (vis[:,:,None] * ax3[:,:,:]).sum(dim=0)
            out_texture[image_jis[:,0],image_jis[:,1],:] = (vis[:,:,None] * texture_samples).sum(dim=0)

            if compute_reg:
                s0 = torch.cat([torch.zeros_like(alpha[:1,:]), (vis)[:-1,:]], dim=0).cumsum(dim=0)
                s1 = torch.cat([torch.zeros_like(alpha[:1,:]), (vis * t)[:-1,:]], dim=0).cumsum(dim=0)
                s2 = torch.cat([torch.zeros_like(alpha[:1,:]), (vis * t * t)[:-1,:]], dim=0).cumsum(dim=0)
                out_reg[image_jis[:,0],image_jis[:,1]] = (vis * (t * t * s0 + s2 - 2 * t * s1)).sum(dim=0)

            if depth_mode == 3:
                visited_mask = (alpha > 1e-3) & (T > 0.5)
                t_view = t[:,:] * view_depth[None,:]
                depth_idx_val, depth_idx = torch.max(
                    visited_mask.int() * (1 + torch.arange(t.shape[0], device=t.device))[:,None],
                    dim=0
                )
                
                out_depth[image_jis[depth_idx_val != 0,0],image_jis[depth_idx_val != 0,1]] = t_view[
                    depth_idx,
                    torch.arange(t.shape[1], device=depth_idx.device)
                ][depth_idx_val != 0]
            final_Ts[image_jis[:,0],image_jis[:,1]] = 1 - acc_alpha
            final_idx[image_jis[:,0],image_jis[:,1]] = idx
    out_img = out_img.swapaxes(0, 1)
    out_depth = out_depth.swapaxes(0, 1)
    out_reg = out_reg.swapaxes(0, 1)
    out_texture = out_texture.swapaxes(0, 1)
    out_normal = out_normal.swapaxes(0, 1)
    final_Ts = final_Ts.swapaxes(0, 1)
    final_idx = final_idx.swapaxes(0, 1)
    return (
        out_img, out_depth, out_reg, out_texture, out_normal, final_Ts, final_idx
    )