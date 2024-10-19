# GStex CUDA Rasterizer

<a href="https://compimaging.dgp.toronto.edu/">
<img src="./assets/logo.gif" width="25%" height="25%"/>
</a>

This repository contains the implementation for the CUDA rasterizer used by [GStex](https://github.com/victor-rong/GStex).

## Installation

While this repository is meant to be used by [GStex](https://github.com/victor-rong/GStex), it is also functional stand-alone.

```
git clone https://github.com/victor-rong/GStex_cuda.git --recursive
conda create --name GStex_cuda -y python=3.8
conda activate GStex_cuda
```

[Pytorch](https://pytorch.org/get-started/locally/) and CUDA toolkit must be installed first. Below, we give an example script for CUDA 12.1, which we tested with. You may need to use a different version depending on your hardware. 

```
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121
conda install -c "nvidia/label/cuda-12.1.0" cuda-toolkit
```

Once this is complete, you can install from source.

```bash
cd GStex_cuda
pip install --upgrade pip setuptools
pip install -e .
```

## Example

We provide a script for training textured Gaussians to overfit to an image. A gif of the training is saved to `./renders/training.gif`. Note that the reported timings may be off the first time you run it as the CUDA code is being compiled.

```bash
python example.py
```

Our implementation includes the L2 depth distortion and normal regularization suggested by [2DGS](https://surfsplatting.github.io/). By setting the number of texels to 0, we have an equivalent 2DGS rasterizer (although the behaviour at numerical edge cases may differ from the official implementation).

```bash
python example.py --num_points 10000 --num_texels 0
```

This script can also be used to check our CUDA implementation against a slower Pytorch implementation.

```bash
python example.py --height 32 --width 32 --num_points 10 --iterations 10 --torch_compare True
```

## Acknowledgements

This codebase is directly built on the [gsplat](https://docs.gsplat.studio/main/) codebase by Vickie Ye, Ruilong Li, and [many other amazing people](https://github.com/nerfstudio-project/gsplat?tab=readme-ov-file#development-and-contribution) from the Nerfstudio team. Much of the method was based on [2DGS](https://surfsplatting.github.io/) by Binbin Huang et al.

## Citation

If you find this repository useful in your projects or papers, please consider citing our paper:
```
@article{rong2024gstex,
  title={GStex: Per-Primitive Texturing of 2D Gaussian Splatting for Decoupled Appearance and Geometry Modeling},
  author={Rong, Victor and Chen, Jingxiang and Bahmani, Sherwin and Kutulakos, Kiriakos N and Lindell, David B},
  journal={arXiv preprint arXiv:2409.12954},
  year={2024}
}
```
