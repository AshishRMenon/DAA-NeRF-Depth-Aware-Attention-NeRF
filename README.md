
# 🔬 Grouped-Conv NeRF: Depth-Aware Rendering with Learnable Bins

This repository implements a novel NeRF-inspired architecture that performs **view-consistent rendering via grouped convolutions** over learnable 3D samples and ray directions.

## 🧠 Key Ideas

- 🔹 **Learnable Depth Bins**: The model predicts `N=64` depth bins per pixel using a CNN on positional-encoded ray origin and direction.
- 🔹 **Grouped Convolutions**: Sampled 3D points (`x = rays_o + t * rays_d`) and ray directions are processed using grouped convolutions per depth bin.
- 🔹 **RGB & Weight Heads**: Each bin outputs RGB and an attention weight. These are fused via a softmax-weighted sum to obtain the final RGB.



## 📦 Files

- `main.py` – training entry point
- `custom_nerf_model.py` – model architecture
- `nerf_dataloader.py` – NeRF-style JSON + image patch dataloader
- `README.md` – you're here!

## 🚀 Highlights

- Patch-wise training with 32x32 crops
- End-to-end depth learning
- Fully convolutional and efficient
