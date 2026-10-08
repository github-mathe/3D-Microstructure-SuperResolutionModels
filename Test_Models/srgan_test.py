import os
import re
from glob import glob
import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split, Subset
from tifffile import imread, imwrite
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import numpy as np
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import scipy.ndimage
import random
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class SuperResolutionDataset(Dataset):
    def __init__(
        self,
        low_res_dir: str,
        high_res_dir: str,
        crop_size=(32, 32, 32),
        upscale_factor: int = 2,
        num_crops: int = 20,
        augment: bool = True,
    ):
        """
        A Dataset for 3D super-resolution that:
          - Preloads volumes into memory as tensors
          - Performs aligned 3D random crops in pure PyTorch
          - Applies optional GPU-friendly augmentations
          - Normalizes each volume to [0,1]
        """
        self.low_res_paths = sorted(
            os.path.join(low_res_dir, f)
            for f in os.listdir(low_res_dir)
            if f.endswith(".tif")
        )
        self.high_res_paths = sorted(
            os.path.join(high_res_dir, f)
            for f in os.listdir(high_res_dir)
            if f.endswith(".tif")
        )
        assert len(self.low_res_paths) == len(self.high_res_paths), \
            "Low-/high-res counts must match"

        self.crop_size = crop_size
        self.upscale_factor = upscale_factor
        self.num_crops = num_crops
        self.augment = augment

        self.volumes_lr = []
        self.volumes_hr = []
        for lr_path, hr_path in zip(self.low_res_paths, self.high_res_paths):
            lr_np = imread(lr_path).astype(np.float32)
            hr_np = imread(hr_path).astype(np.float32)
            if lr_np.max() > lr_np.min():
                lr_np = (lr_np - lr_np.min()) / (lr_np.max() - lr_np.min())
            if hr_np.max() > hr_np.min():
                hr_np = (hr_np - hr_np.min()) / (hr_np.max() - hr_np.min())
            lr_t = torch.from_numpy(lr_np).unsqueeze(0)
            hr_t = torch.from_numpy(hr_np).unsqueeze(0)
            self.volumes_lr.append(lr_t)
            self.volumes_hr.append(hr_t)

        self.data_index = [
            (vol_idx, crop_idx)
            for vol_idx in range(len(self.volumes_lr))
            for crop_idx in range(self.num_crops)
        ]

    def __len__(self):
        return len(self.data_index)

    def random_crop(self, img_lr: torch.Tensor, img_hr: torch.Tensor):
        _, D, H, W = img_lr.shape
        cd, ch, cw = self.crop_size
        r = self.upscale_factor

        d0 = torch.randint(0, D - cd + 1, ())
        h0 = torch.randint(0, H - ch + 1, ())
        w0 = torch.randint(0, W - cw + 1, ())

        lr_crop = img_lr[:, d0:d0+cd, h0:h0+ch, w0:w0+cw]
        hr_crop = img_hr[:, d0*r:d0*r+cd*r, h0*r:h0*r+ch*r, w0*r:w0*r+cw*r]
        return lr_crop, hr_crop

    def augment_data(self, img_lr: torch.Tensor, img_hr: torch.Tensor):
        if torch.rand(()) > 0.5:
            img_lr = torch.flip(img_lr, dims=[2]); img_hr = torch.flip(img_hr, dims=[2])
        if torch.rand(()) > 0.5:
            img_lr = torch.flip(img_lr, dims=[3]); img_hr = torch.flip(img_hr, dims=[3])
        if torch.rand(()) > 0.5:
            img_lr = torch.flip(img_lr, dims=[1]); img_hr = torch.flip(img_hr, dims=[1])
        k = int(torch.randint(0, 4, (1,)))
        img_lr = torch.rot90(img_lr, k, dims=[2,3]); img_hr = torch.rot90(img_hr, k, dims=[2,3])
        return img_lr, img_hr

    def __getitem__(self, idx: int):
        vol_idx, _ = self.data_index[idx]
        lr_vol = self.volumes_lr[vol_idx]
        hr_vol = self.volumes_hr[vol_idx]

        lr_crop, hr_crop = self.random_crop(lr_vol, hr_vol)
        if self.augment:
            lr_crop, hr_crop = self.augment_data(lr_crop, hr_crop)

        return lr_crop, hr_crop



class PixelShuffle3D(nn.Module):
    def __init__(self, upscale_factor):
        super(PixelShuffle3D, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x):
        batch_size, channels, depth, height, width = x.size()
        r = self.upscale_factor
        channels //= r**3
        x = x.view(batch_size, channels, r, r, r, depth, height, width)
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4)
        return x.reshape(batch_size, channels, depth * r, height * r, width * r)


class ResidualBlock3D(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)
        return x + residual  # Skip connection inside residual block
    
class UpsampleBLock3D(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBLock3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, in_channels * up_scale ** 3, kernel_size=3, padding=1)
        self.pixel_shuffle = PixelShuffle3D(up_scale)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x
    
class Generator3D(nn.Module):
    def __init__(self, upscale_factor, num_residual_block=16):
        super(Generator3D, self).__init__()
        upsample_block_num = int(math.log(upscale_factor, 2))

        self.block1 = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        
        self.res_blocks = nn.ModuleList([ResidualBlock3D(64) for _ in range(num_residual_block)])
        self.conv2 = nn.Conv3d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(64)
        
        upsampling_blocks = [UpsampleBLock3D(64, 2) for _ in range(upsample_block_num)]
        upsampling_blocks.append(nn.Conv3d(64, 1, kernel_size=9, padding=4))
        self.upsampling = nn.Sequential(*upsampling_blocks)

        self.activation = nn.Sigmoid()

    def forward(self, x):
        x1 = self.block1(x)
        res_out = x1
        
        for res_block in self.res_blocks:
            res_out = res_block(res_out) + res_out  # Skip connection
        
        x3 = self.conv2(res_out)
        x3 = self.bn2(x3)
        x4 = self.upsampling(x1 + x3)
        return self.activation(x4)


def test_super_resolution(model_path, low_res_path, base_output_dir, upscale_factor=4, num_residual_block=4):
    import torch.nn.functional as F

    # Initialize the model
    model = Generator3D(upscale_factor=upscale_factor, num_residual_block=num_residual_block)
    state_dict = torch.load(model_path, map_location=torch.device("cpu"))
    if any(key.startswith("module.") for key in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()

    # Use CPU to avoid CuDNN issues
    # device = torch.device("cpu")
    model = model.to(device)

    # Load and normalize input
    low_res = imread(low_res_path).astype(np.float32)
    if low_res.max() > low_res.min():
        low_res = (low_res - low_res.min()) / (low_res.max() - low_res.min())

    low_res_tensor = torch.from_numpy(low_res).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, D, H, W)
    _, _, D, H, W = low_res_tensor.shape

    # Compute padding needed for each dim to be divisible by 2^n
    pad_factor = 2 ** int(np.log2(upscale_factor))
    pad_D = (pad_factor - D % pad_factor) % pad_factor
    pad_H = (pad_factor - H % pad_factor) % pad_factor
    pad_W = (pad_factor - W % pad_factor) % pad_factor

    # Apply padding at the end (right/bottom)
    padding = (0, pad_W, 0, pad_H, 0, pad_D)  # (W_left, W_right, H_top, H_bottom, D_front, D_back)
    low_res_padded = F.pad(low_res_tensor, padding, mode='reflect')

    # Predict
    with torch.no_grad():
        high_res_pred = model(low_res_padded).squeeze(0).squeeze(0).cpu()

    # Remove extra padding (scaled by upscale factor)
    D_out, H_out, W_out = D * upscale_factor, H * upscale_factor, W * upscale_factor
    high_res_pred = high_res_pred[:D_out, :H_out, :W_out]

    # Save output
    output_dir = os.path.join(base_output_dir, f"x{upscale_factor}_NB_{num_residual_block}_outputs")
    os.makedirs(output_dir, exist_ok=True)

    original_filename = os.path.basename(low_res_path)
    new_filename = re.sub(r"_(\d+)_", lambda m: f"_{int(m.group(1)) * upscale_factor}_", original_filename)
    name, ext = os.path.splitext(new_filename)
    final_filename = f"{name}_NB{num_residual_block}{ext}"
    save_path = os.path.join(output_dir, final_filename)

    imwrite(save_path, (high_res_pred.numpy() * 255).astype(np.uint8))
    print(f"✅ Saved: {save_path}")


# Example Usage
if __name__ == "__main__":
    upscale_factor = 2
    num_residual_block = 16

    model_path = os.path.join(PROJECT_ROOT, "checkpoints", "SRGAN3D", f"model_x{upscale_factor}", f"sr_generator_x{upscale_factor}.pth")
    input_dir = os.path.join(PROJECT_ROOT, "data", "Testing_Data", "100")
    output_base_dir = os.path.join(PROJECT_ROOT, "data", "SR_Data", "SRGAN3D")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    tif_files = sorted(glob(os.path.join(input_dir, "*.tif")))
    if not tif_files:
        raise RuntimeError(f"No .tif files found in {input_dir}")

    for tif_path in tif_files:
        test_super_resolution(model_path, tif_path, output_base_dir, upscale_factor, num_residual_block)


