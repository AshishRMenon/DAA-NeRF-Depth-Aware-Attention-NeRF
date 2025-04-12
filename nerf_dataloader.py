import torch
import json
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
def load_blender_json(json_path, img_dir, train=True, downsample=1.0):
    with open(json_path, 'r') as f:
        meta = json.load(f)

    imgs = []
    poses = []
    for frame in meta['frames']:
        if train==True:
            img_path = os.path.join(img_dir, "lego/train",os.path.basename(frame['file_path']) + '.png')
        else:
            img_path = os.path.join(img_dir, "lego/test",os.path.basename(frame['file_path']) + '.png')

        if os.path.exists(img_path):
            img = Image.open(img_path).convert('RGB')
            if downsample != 1.0:
                W, H = img.size
                img = img.resize((int(W * downsample), int(H * downsample)), Image.LANCZOS)
            imgs.append(transforms.ToTensor()(img).permute(1, 2, 0))  # (H, W, 3)
            poses.append(np.array(frame['transform_matrix']))

    imgs = torch.stack(imgs)  # (N, H, W, 3)
    poses = torch.tensor(poses, dtype=torch.float32)  # (N, 4, 4)

    return imgs, poses, meta['camera_angle_x']





def get_ray_directions(H, W, focal):
    i, j = torch.meshgrid(torch.arange(W), torch.arange(H), indexing='xy')
    dirs = torch.stack([(i - W * 0.5) / focal,
                        -(j - H * 0.5) / focal,
                        -torch.ones_like(i)], -1)  # (H, W, 3)
    return dirs

def get_rays(directions, c2w):
    rays_d = directions @ c2w[:3, :3].T  # (H, W, 3)
    rays_o = c2w[:3, 3].expand_as(rays_d)  # (H, W, 3)
    return rays_o, rays_d

class NeRFRayPatchDataset(Dataset):
    def __init__(self, json_path, img_dir, patch_size=32, downsample=1.0):
        super().__init__()
        self.imgs, self.poses, self.camera_angle_x = load_blender_json(json_path, img_dir, downsample)
        self.patch_size = patch_size
        self.H, self.W = self.imgs.shape[1:3]

        # Compute focal length
        self.focal = 0.5 * self.W / np.tan(0.5 * self.camera_angle_x)

        # Precompute rays for all images
        directions = get_ray_directions(self.H, self.W, self.focal)
        self.rays = []
        self.pixels = []

        for i in range(len(self.imgs)):
            rays_o, rays_d = get_rays(directions, self.poses[i])
            self.rays.append((rays_o, rays_d))
            self.pixels.append(self.imgs[i])

        # Build list of patch locations: (img_idx, i, j)
        self.patches = []
        for img_idx in range(len(self.imgs)):
            for i in range(0, self.H - patch_size + 1, patch_size):
                for j in range(0, self.W - patch_size + 1, patch_size):
                    self.patches.append((img_idx, i, j))

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        img_idx, i, j = self.patches[idx]
        rays_o, rays_d = self.rays[img_idx]
        rgb = self.pixels[img_idx]

        rays_o_patch = rays_o[i:i+self.patch_size, j:j+self.patch_size]
        rays_d_patch = rays_d[i:i+self.patch_size, j:j+self.patch_size]
        rgb_patch = rgb[i:i+self.patch_size, j:j+self.patch_size]

        return {
            'rays_o': rays_o_patch,  # (P, P, 3)
            'rays_d': rays_d_patch,
            'gt_rgb': rgb_patch
        }


def get_patch_dataloader(json_path, img_dir, patch_size=32, downsample=1.0, batch_size=32, shuffle=True, num_workers=4):
    dataset = NeRFRayPatchDataset(
        json_path=json_path,
        img_dir=img_dir,
        patch_size=patch_size,
        downsample=downsample
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    return loader


