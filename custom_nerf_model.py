import torch
import torch.nn as nn
import torch.nn.functional as F

def positional_encoding(x, num_freqs=10):
    enc = [x]
    for i in range(num_freqs):
        for fn in [torch.sin, torch.cos]:
            enc.append(fn((2.0 ** i) * x))
    return torch.cat(enc, dim=-1)

class DepthPredictionModule(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_bins=64):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, num_bins, 1)
        )

    def forward(self, x, z_near=2.0, z_far=6.0):
        depths = self.predictor(x)  # (B, num_bins, H, W)
        depths = torch.sigmoid(depths) * (z_far - z_near) + z_near
        return depths

class GroupedConvNeRF(nn.Module):
    def __init__(self, num_bins=64, num_freqs=10, z_near=2.0, z_far=6.0):
        super().__init__()
        self.num_bins = num_bins
        self.z_near = z_near
        self.z_far = z_far
        self.num_freqs = num_freqs

        pe_dim = 3 * (2 * num_freqs + 1)
        self.depth_net = DepthPredictionModule(in_channels=2 * pe_dim, hidden_channels=64, num_bins=num_bins)

        self.group_conv = nn.Sequential(
            nn.Conv1d(num_bins * 6, num_bins * 6, 1, groups=num_bins),
            nn.ReLU(),
            nn.Conv1d(num_bins * 6, num_bins * 32, 1, groups=num_bins),
            nn.ReLU()
        )

        self.rgb_head = nn.Conv1d(num_bins * 32, num_bins * 3, 1, groups=num_bins)
        self.weight_head = nn.Conv1d(num_bins * 32, num_bins * 1, 1, groups=num_bins)

    def forward(self, rays_o, rays_d):
        B, _, H, W = rays_o.shape

        pe_o = positional_encoding(rays_o.permute(0, 2, 3, 1), self.num_freqs)
        pe_d = positional_encoding(rays_d.permute(0, 2, 3, 1), self.num_freqs)
        pe_input = torch.cat([pe_o, pe_d], dim=-1).permute(0, 3, 1, 2)

        depths = self.depth_net(pe_input, self.z_near, self.z_far)  # (B, D, H, W)

        rays_o = rays_o.unsqueeze(1)
        rays_d = rays_d.unsqueeze(1)
        depths = depths.unsqueeze(2)
        pts = rays_o + depths * rays_d  # (B, D, 3, H, W)

        rays_d_exp = rays_d.expand(-1, self.num_bins, -1, -1, -1)
        all_inputs = torch.cat([pts, rays_d_exp], dim=2)
        all_inputs = all_inputs.permute(0, 3, 4, 1, 2).reshape(B, H * W, self.num_bins * 6).permute(0, 2, 1)

        features = self.group_conv(all_inputs)  # (B, D*32, H*W)

        rgb = self.rgb_head(features)  # (B, D*3, H*W)
        weights = self.weight_head(features)  # (B, D, H*W)

        rgb = rgb.view(B, self.num_bins, 3, H, W).permute(0, 2, 1, 3, 4)  # (B, 3, H, W, D)
        weights = weights.view(B, self.num_bins, H, W)
        weights = F.softmax(weights, dim=1).unsqueeze(1)  # (B, 1, D, H, W)

        rgb_out = (rgb * weights).sum(dim=2)  # (B, 3, H, W)
        return rgb_out

