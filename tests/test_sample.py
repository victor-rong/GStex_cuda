import pytest
import torch
import numpy as np
from gstex_cuda import _torch_impl as _T
from gstex_cuda.texture_sample import texture_sample

@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.order(1)
def test_sample():
    device = torch.device("cuda")
    sz = 5
    texture_dims = torch.stack([
        torch.randint(6, size=(sz,), dtype=torch.int32) + 2,
        torch.randint(7, size=(sz,), dtype=torch.int32) + 2,
        torch.zeros((sz,), dtype=torch.int32)
    ], dim=-1
    ).to(device)
    hws = texture_dims[:,0] * texture_dims[:,1]
    texture_dims[:,-1] = torch.cumsum(hws, dim=0) - hws
    total_sz = torch.sum(hws).item()
    num_channels = 10

    texture = torch.rand((total_sz, num_channels,)).to(device)
    num_queries = 100
    uvs = torch.rand((num_queries, 2)).to(device)

    texture_info = (sz, 1, num_channels)
    ids = torch.randint(sz, size=(num_queries,))
    query_texture_dims = texture_dims[ids,:]

    fast_outputs = texture_sample(texture_info, query_texture_dims, texture, uvs)
    torch_outputs = _T.sample_texture(query_texture_dims, texture, uvs)

    torch.testing.assert_close(fast_outputs, torch_outputs)

if __name__ == "__main__":
    test_sample()