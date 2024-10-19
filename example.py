import math
import os
from pathlib import Path
from typing import Optional
import argparse

import numpy as np
import torch
from gstex_cuda.texture import texture_gaussians
from gstex_cuda.get_aabb_2d import get_aabb_2d, get_num_tiles_hit_2d, project_points
from gstex_cuda._torch_impl import normalized_quat_to_rotmat
from PIL import Image
from torch import Tensor, optim
from gstex_cuda.timer import Timer

def seed_everything(seed: int):
    import random
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def write_imgs(imgs, dim, out_dir, filename):
    frames = []
    for i in range(len(imgs)):
        img_h = imgs[i][0].shape[0]
        img_w = imgs[i][0].shape[1]
        concat_img = np.zeros((dim[0]*img_h, dim[1]*img_w, 3,), dtype=imgs[i][0].dtype)
        for j in range(len(imgs[i])):
            xi = j % dim[0]
            xj = j // dim[0]
            concat_img[xi*img_h:(xi+1)*img_h,xj*img_w:(xj+1)*img_w] = imgs[i][j]
        frames.append(concat_img)
    frames = [Image.fromarray(frame) for frame in frames]
    frames[0].save(
        f"{out_dir}/{filename}",
        save_all=True,
        append_images=frames[1:],
        optimize=False,
        duration=5,
        loop=0,
    )

class SimpleTrainer:
    """Trains random gaussians to fit an image."""

    def __init__(
        self,
        gt_image: Tensor,
        num_points: int = 2000,
        num_texels: int = 1000000,
    ):
        self.device = torch.device("cuda:0")
        self.gt_image = gt_image.to(device=self.device)
        self.num_points = num_points
        self.num_texels = num_texels

        fov_x = math.pi / 2.0
        self.H, self.W = gt_image.shape[0], gt_image.shape[1]
        self.focal = 0.5 * float(self.W) / math.tan(0.5 * fov_x)
        self.img_size = torch.tensor([self.W, self.H, 1], device=self.device)

        self._init_gaussians()

    def _init_gaussians(self):
        """Initialize gaussians and texels"""
        self.means = torch.rand(self.num_points, 3, device=self.device) - 0.5
        self.means[:,:2] *= 16
        self.scales = 0.5 * np.log(1/self.num_points) * torch.rand(self.num_points, 3, device=self.device)
        d = 3
        self.rgbs = torch.rand(self.num_points, d, device=self.device)

        u = torch.rand(self.num_points, 1, device=self.device)
        v = torch.rand(self.num_points, 1, device=self.device)
        w = torch.rand(self.num_points, 1, device=self.device)

        self.quats = torch.cat(
            [
                torch.sqrt(1.0 - u) * torch.sin(2.0 * math.pi * v),
                torch.sqrt(1.0 - u) * torch.cos(2.0 * math.pi * v),
                torch.sqrt(u) * torch.sin(2.0 * math.pi * w),
                torch.sqrt(u) * torch.cos(2.0 * math.pi * w),
            ],
            -1,
        )
        # Set texture dimensions so that there are around num_texels texels
        self.th = int(np.sqrt(self.num_texels / self.num_points) + 1)
        self.tw = int(np.sqrt(self.num_texels / self.num_points) + 1)
        self.opacities = torch.ones((self.num_points, 1), device=self.device)
        self.mapping = torch.zeros((self.num_points, 1, 4), device=self.device)
        self.mapping[:,:,:2] = 0.5
        scale = 0.25 * np.sqrt(1 / torch.sum(torch.exp(self.scales[:,0] + self.scales[:,1])).item())
        self.mapping[:,:,2] = np.log(scale)
        self.mapping[:,:,3] = 2.0 * math.pi * torch.rand(self.num_points, 1, device=self.device)
        self.texture = torch.rand(self.num_points*self.th*self.tw, 3, device=self.device)
        self.viewmat = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 8.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            device=self.device,
        )
        self.c2w = self.viewmat.inverse()
        self.background = torch.zeros(d, device=self.device)

        self.means.requires_grad = True
        self.scales.requires_grad = True
        self.quats.requires_grad = True
        self.rgbs.requires_grad = True
        self.opacities.requires_grad = True
        self.viewmat.requires_grad = False
        self.mapping.requires_grad = False
        self.texture.requires_grad = True

    def forward(self, timer, use_torch_impl=False):
        """
        Perform a render pass
        """
        timer.start("preprocess")
        scales = torch.zeros_like(self.scales)
        scales[:,:2] = torch.exp(self.scales[:,:2])
        scales[:,-1] = 1e-5 * torch.mean(scales[:,:-1], dim=-1).detach()
        quats = self.quats / self.quats.norm(dim=-1, keepdim=True)
        Rs = normalized_quat_to_rotmat(quats)
        uv0 = self.mapping[:,:,:2]
        uvscale = torch.exp(self.mapping[:,:,None,2])
        theta = self.mapping[:,:,None,3]
        ax1 = Rs[:,None,:,0]
        ax2 = Rs[:,None,:,1]
        umap = uvscale * (ax1 * torch.cos(theta) + ax2 * torch.sin(theta))
        vmap = uvscale * (-ax1 * torch.sin(theta) + ax2 * torch.cos(theta))
        
        texture_info = (self.num_points, 1, 3)
        texture_dims = torch.zeros(self.num_points, 3, dtype=torch.int32, device=self.means.device)
        texture_dims[:,0] = self.th
        texture_dims[:,1] = self.tw
        texture_dims[:,2] = torch.cumsum(texture_dims[:,0] * texture_dims[:,1], dim=-1) - texture_dims[:,0] * texture_dims[:,1]
        timer.stop()
        timer.start("project")
        B_SIZE = 16
        intrins = (self.focal, self.focal, self.W/2, self.H/2)
        xys, depths = project_points(self.means, self.viewmat.squeeze()[:3,:], intrins)

        # No splatting is done, the 2D Gaussian's actual screen-space AABB is computed
        centers, extents = get_aabb_2d(self.means, scales, 1, quats, self.viewmat, intrins)
        num_tiles_hit = get_num_tiles_hit_2d(centers, extents, self.H, self.W, B_SIZE)
        timer.stop()
        timer.start("render")
        outputs = texture_gaussians(
            texture_info,
            texture_dims,
            centers,
            extents,
            depths,
            num_tiles_hit,
            torch.sigmoid(self.rgbs),
            torch.sigmoid(self.opacities),
            self.means,
            scales,
            1,
            quats,
            uv0,
            umap,
            vmap,
            torch.sigmoid(self.texture),
            self.viewmat,
            self.c2w,
            self.focal,
            self.focal,
            self.W / 2,
            self.H / 2,
            self.H,
            self.W,
            B_SIZE,
            1<<8,
            self.background,
            use_torch_impl=use_torch_impl
        )
        timer.stop()

        return outputs

    def compute_loss(self, outputs, gt_image):
        """
        Compute the loss
        """
        mse_loss = torch.nn.MSELoss()
        out_rgb = outputs[0][:,:,:]
        out_img = outputs[4][:,:,:3]
        out_depth = outputs[1][:,:,None]
        out_reg = outputs[2][:,:,None]
        out_normal = outputs[5]
        gt_loss = mse_loss(out_img, gt_image)
        # L2 depth regularization
        reg_loss = torch.mean(out_reg)
        # normal loss, encourages xy Gaussians
        normal_loss = torch.mean(
            out_normal[:,:,0]**2 +
            out_normal[:,:,1]**2 +
            (1 - out_normal[:,:,2])**2
        )
        loss = gt_loss + reg_loss + normal_loss
        return loss

    def train(
        self,
        iterations: int = 1000,
        lr: float = 0.01,
        save_imgs: bool = False,
        torch_compare: bool = False,
    ):
        """
        An example training to overfit to a 2D image. Also supports output and gradient comparison against an autograd implementations.
        """
        fast_timer = Timer(disabled=False)
        slow_timer = Timer(disabled=False)
        optimizer = optim.Adam(
            [self.rgbs, self.means, self.scales, self.opacities, self.quats, self.mapping, self.texture], lr
        )
        frames = []
        for iter in range(iterations):
            outputs = []
            torch_outputs = []
            fast_grads = []
            slow_grads = []
            if torch_compare:
                optimizer.zero_grad()
                torch_outputs = self.forward(slow_timer, use_torch_impl=True)

                loss = self.compute_loss(torch_outputs, self.gt_image)
                slow_timer.start("backward")
                loss.backward()
                slow_timer.stop()

                slow_grads = [
                    self.means.grad.detach() if self.means.grad is not None else torch.zeros_like(self.means),
                    self.scales.grad.detach() if self.scales.grad is not None else torch.zeros_like(self.scales),
                    self.quats.grad.detach() if self.quats.grad is not None else torch.zeros_like(self.quats),
                    self.rgbs.grad.detach() if self.rgbs.grad is not None else torch.zeros_like(self.rgbs),
                    self.opacities.grad.detach() if self.opacities.grad is not None else torch.zeros_like(self.opacities),
                    self.mapping.grad.detach() if self.mapping.grad is not None else torch.zeros_like(self.mapping),
                    self.texture.grad.detach() if self.texture.grad is not None else torch.zeros_like(self.texture),
                ]

            optimizer.zero_grad()
            outputs = self.forward(fast_timer, use_torch_impl=False)
            if not torch_compare:
                torch_outputs = outputs
            
            loss = self.compute_loss(outputs, self.gt_image)
            fast_timer.start("backward")
            loss.backward()
            fast_timer.stop()
            fast_grads = [
                self.means.grad.detach() if self.means.grad is not None else torch.zeros_like(self.means),
                self.scales.grad.detach() if self.scales.grad is not None else torch.zeros_like(self.scales),
                self.quats.grad.detach() if self.quats.grad is not None else torch.zeros_like(self.quats),
                self.rgbs.grad.detach() if self.rgbs.grad is not None else torch.zeros_like(self.rgbs),
                self.opacities.grad.detach() if self.opacities.grad is not None else torch.zeros_like(self.opacities),
                self.mapping.grad.detach() if self.mapping.grad is not None else torch.zeros_like(self.mapping),
                self.texture.grad.detach() if self.texture.grad is not None else torch.zeros_like(self.texture),
            ]

            if torch_compare:
                for slow_output, fast_output in zip(torch_outputs, outputs):
                    torch.testing.assert_close(fast_output, slow_output)

                for slow_grad, fast_grad in zip(slow_grads, fast_grads):
                    torch.testing.assert_close(fast_grad, slow_grad)

            fast_timer.start("step")
            optimizer.step()
            fast_timer.stop()

            print(f"Iteration {iter + 1}/{iterations}, Loss: {loss.item()}")

            if save_imgs and iter % 10 == 0:
                out_img = outputs[4][:,:,:3]
                out_depth = outputs[1][:,:,None]
                frames.append(
                    [
                        (out_img.detach().cpu().numpy() * 255).astype(np.uint8),
                    ]
                )
        if save_imgs:
            out_dir = os.path.join(os.getcwd(), "renders")
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            write_imgs(frames,(1, 1), out_dir, "training.gif")
        print("Fast times")
        fast_timer.dump()
        
        if torch_compare:
            print("Slow times")
            slow_timer.dump()

def image_path_to_tensor(image_path: Path):
    import torchvision.transforms as transforms

    img = Image.open(image_path)
    transform = transforms.ToTensor()
    img_tensor = transform(img).permute(1, 2, 0)[..., :3]
    return img_tensor

def main(
    height: int = 256,
    width: int = 256,
    num_points: int = 100,
    num_texels: int = 1000000,
    save_imgs: bool = True,
    torch_compare: bool = False,
    img_path: Optional[Path] = None,
    iterations: int = 1000,
    lr: float = 1e-2,
) -> None:
    """
        height: Height of GT image if default image is used
        width: Width of GT image if default image is used
        num_points: Number of Gaussians
        num_texels: Approximate number of texels. Set to 0 if you want 2DGS behaviour.
        save_imgs: Whether to save as a gif
        torch_compare: Whether to compare against the torch implementation. As the torch implementation is slow,
                       you should set small parameters (e.g. 10 points, 10 iterations32 x 32 GT). Note that there
                       are actually a few small discrepancies between the torch and CUDA implementation, but for
                       < 10 iterations, these generally don't affect the renders and gradients.
        img_path: Path to an image to overfit to. If set to None, uses default red/blue/white image.
        iterations: Number of iterations.
        lr: Learning rate. All parameters are set to have this learning rate with Adam optimization.
    """
    seed_everything(1)
    if img_path:
        gt_image = image_path_to_tensor(img_path)
    else:
        gt_image = torch.ones((height, width, 3)) * 1.0
        # make top left and bottom right red, blue
        gt_image[: height // 2, : width // 2, :] = torch.tensor([1.0, 0.0, 0.0])
        gt_image[height // 2 :, width // 2 :, :] = torch.tensor([0.0, 0.0, 1.0])

    trainer = SimpleTrainer(gt_image=gt_image, num_points=num_points, num_texels=num_texels)
    trainer.train(
        iterations=iterations,
        lr=lr,
        save_imgs=save_imgs,
        torch_compare=torch_compare,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--height", default=256, type=int)
    parser.add_argument("--width", default=256, type=int)
    parser.add_argument("--num_points", default=100, type=int)
    parser.add_argument("--num_texels", default=1000000, type=int)
    parser.add_argument("--iterations", default=1000, type=int)
    parser.add_argument("--img_path", default="", type=str)
    parser.add_argument("--save_imgs", default=True, type=bool)
    parser.add_argument("--torch_compare", default=False, type=bool)
    args = parser.parse_args()
    main(
        height=args.height,
        width=args.width,
        num_points=args.num_points,
        num_texels=args.num_texels,
        save_imgs=args.save_imgs,
        torch_compare=args.torch_compare,
        img_path=None if args.img_path == "" else Path(args.img_path),
        iterations=args.iterations,
    )
