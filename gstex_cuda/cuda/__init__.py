from typing import Callable


def _make_lazy_cuda_func(name: str) -> Callable:
    def call_cuda(*args, **kwargs):
        # pylint: disable=import-outside-toplevel
        from ._backend import _C

        return getattr(_C, name)(*args, **kwargs)

    return call_cuda


compute_sh_forward = _make_lazy_cuda_func("compute_sh_forward")
compute_sh_backward = _make_lazy_cuda_func("compute_sh_backward")
map_gaussian_to_intersects = _make_lazy_cuda_func("map_gaussian_to_intersects")
get_tile_bin_edges = _make_lazy_cuda_func("get_tile_bin_edges")
get_aabb_2d = _make_lazy_cuda_func("get_aabb_2d")

# GStex
texture_forward = _make_lazy_cuda_func("texture_forward")
texture_backward = _make_lazy_cuda_func("texture_backward")

texture_sample_forward = _make_lazy_cuda_func("texture_sample_forward")
texture_sample_backward = _make_lazy_cuda_func("texture_sample_backward")

texture_edit = _make_lazy_cuda_func("texture_edit")