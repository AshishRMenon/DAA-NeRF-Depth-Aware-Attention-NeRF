import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import imageio
import os, json
from huggingface_hub import snapshot_download
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
from skimage import io

from nerf_dataloader import get_patch_dataloader
from custom_nerf_model import GroupedConvNeRF
import random



def test(H,W,camera_angle_x,model,device):

    rendered_images = []
    psnr_scores = []
    ssim_scores = []

    H, W = train_imgs[0].shape[:2]
    focal = 0.5 * W / np.tan(0.5 * camera_angle_x)
    model.eval()
    with torch.no_grad():
        for n,data in tqdm(range(0, len(test_imgs), 2)):
            gt_img = test_imgs[i][:,:,:3]
            pose = torch.tensor(test_poses[i], dtype=torch.float32).to(device)
            rays_o, rays_d = get_ray_bundle(H, W, focal, pose,device)
            rays_o, rays_d = rays_o.unsqueeze(0), rays_d.unsqueeze(0)
            pred_img = model(rays_o, rays_d).squeeze(0).permute(1, 2, 0).cpu().numpy()
            pred_img = np.clip(pred_img, 0, 1)
            rendered_images.append(pred_img)
            psnr_scores.append(psnr(gt_img, pred_img, data_range=1.0))
            ssim_scores.append(ssim(gt_img, pred_img, channel_axis=-1, data_range=1.0,multichannel=True))
            os.makedirs('./rendered_op_learnable_depth_nerf/',exist_ok=True)
            io.imsave('./rendered_op_learnable_depth_nerf/out_{}.png'.format(i),np.uint8(np.clip(pred_img*255.0,0,255)))


    print(f"Average PSNR: {np.mean(psnr_scores):.2f} dB")
    print(f"Average SSIM: {np.mean(ssim_scores):.3f}")

def train(train_loader,model):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    ckpt = {}

    for epoch in range(0,500,1):
        model.train()
        total_loss = 0
        optimizer.zero_grad()
        psnr_scores = []
        ssim_scores = []
        for n,data in enumerate(train_loader):
            rays_o,rays_d,image = data['rays_o'].permute(0,3,1, 2), data['rays_d'].permute(0,3,1, 2), data['gt_rgb'].permute(0,3,1, 2)

            pred = model(rays_o.to(device), rays_d.to(device))
            loss = criterion(pred, image.to(device))
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            pred_np = pred.clone().detach().permute(0,2, 3, 1).cpu().numpy()
            image_np = image.clone().detach().permute(0,2, 3, 1).cpu().numpy()
            for b in range(pred.shape[0]):
                psnr_scores.append(psnr(image_np[b], np.clip(pred_np[b], 0, 1), data_range=1.0))
                ssim_scores.append(ssim(image_np[b], np.clip(pred_np[b], 0, 1), channel_axis=-1, data_range=1.0,multichannel=True))

        print("loss for epoch :{} is :{}".format(epoch,total_loss/100.0))
        print("epoch:{}, psnr :{}, ssim :{}".format(epoch,np.mean(psnr_scores),np.mean(ssim_scores)))

        # test(H,W,camera_angle_x,model,device)
        ckpt['model'] = model
        ckpt['model_state'] = model.state_dict()
        ckpt['optimizer'] = optimizer.state_dict()
        torch.save(ckpt, f'./model_ckpt_pe/nerf_model_epoch_{epoch+1}.pth')

    return model

if __name__=='__main__':


    dataset_path = './dataset/nerf_synthetic'
    ckpt_path = './model_ckpt_pe'
    train_loader = get_patch_dataloader(json_path='./dataset/nerf_synthetic/lego/transforms_train.json',img_dir=dataset_path)
    test_loader = get_patch_dataloader(json_path='./dataset/nerf_synthetic/lego/transforms_test.json',img_dir = dataset_path)

    # os.makedirs(dataset_path,exist_ok=True)
    # os.makedirs(ckpt_path,exist_ok=True)
    # dataset_path = snapshot_download(repo_id="phuckstnk63/nerf-synthetic", local_dir = dataset_path ,repo_type="dataset")
    # print("Downloaded to:", dataset_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model = DepthAwareNeRF(num_bins=64)
    model = GroupedConvNeRF()
    # model.load_state_dict(torch.load('./model_ckpt_2/nerf_model_epoch_500.pth')['model_state'])

    # print(model.state_dict()['depth_bins'])
    # model = torch.load('./model_ckpt_2/nerf_model_epoch_500.pth')['model']
    model = model.to(device)
    model = train(train_loader,model)