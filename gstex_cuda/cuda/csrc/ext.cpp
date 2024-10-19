#include "bindings.h"
#include "get_aabb_2d.cuh"
#include "texture.cuh"
#include "texture_sample.cuh"
#include "texture_edit.cuh"
#include <torch/extension.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // auto diff functions
    m.def("compute_sh_forward", &compute_sh_forward_tensor);
    m.def("compute_sh_backward", &compute_sh_backward_tensor);
    // utils
    m.def("map_gaussian_to_intersects", &map_gaussian_to_intersects_tensor);
    m.def("get_tile_bin_edges", &get_tile_bin_edges_tensor);

    // GStex
    m.def("texture_forward", &texture_forward_tensor);
    m.def("texture_backward", &texture_backward_tensor);
    m.def("get_aabb_2d", &get_aabb_2d_tensor);

    m.def("texture_sample_forward", &texture_sample_forward_tensor);
    m.def("texture_sample_backward", &texture_sample_backward_tensor);

    m.def("texture_edit", &texture_edit_tensor);
}