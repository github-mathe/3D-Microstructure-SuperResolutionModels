import os
from glob import glob
import re
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

# Pixel Shuffle for 3D
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



class ResidualDenseBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, growth_rate=32, num_layers=3):
        super(ResidualDenseBlock3D, self).__init__()
        self.layers = nn.ModuleList()
        
        # ✅ First Conv3D to ensure correct channel expansion
        self.initial_conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.leaky_relu = nn.LeakyReLU(inplace=True)

        channels = out_channels  # ✅ Ensure channel count is correctly initialized

        for _ in range(num_layers):
            self.layers.append(nn.Conv3d(channels, growth_rate, kernel_size=3, padding=1))
            channels += growth_rate

        self.conv_final = nn.Conv3d(channels, out_channels, kernel_size=1)  # ✅ Match original out_channels

    def forward(self, x):
        x = self.leaky_relu(self.initial_conv(x))  # ✅ Ensure first convolution expands channels
        inputs = x

        for layer in self.layers:
            out = self.leaky_relu(layer(inputs))
            inputs = torch.cat((inputs, out), dim=1)  # Expands channel dimensions
        
        return self.conv_final(inputs) + x  # ✅ Ensures residual connection matches output


# Updated 3D Super-Resolution UNet
class SRUNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=64, upscale_factor=2):
        super(SRUNet3D, self).__init__()
        self.upscale_factor = upscale_factor
        features = init_features

        self.encoder1 = ResidualDenseBlock3D(in_channels, features)
        self.downconv1 = nn.Conv3d(features, features * 2, kernel_size=3, stride=2, padding=1)
        self.encoder2 = ResidualDenseBlock3D(features * 2, features * 2)
        self.downconv2 = nn.Conv3d(features * 2, features * 4, kernel_size=3, stride=2, padding=1)
        self.encoder3 = ResidualDenseBlock3D(features * 4, features * 4)
        self.downconv3 = nn.Conv3d(features * 4, features * 8, kernel_size=3, stride=2, padding=1)
        self.bottleneck = ResidualDenseBlock3D(features * 8, features * 8)

        self.upconv3 = nn.ConvTranspose3d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = ResidualDenseBlock3D(features * 4 * 2, features * 4)
        self.upconv2 = nn.ConvTranspose3d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = ResidualDenseBlock3D(features * 2 * 2, features * 2)
        self.upconv1 = nn.ConvTranspose3d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = ResidualDenseBlock3D(features * 2, features)

        self.upconv_final = nn.Conv3d(features, features * (upscale_factor ** 3), kernel_size=3, padding=1)
        self.pixel_shuffle_final = PixelShuffle3D(upscale_factor)

        self.conv_final = nn.Conv3d(features, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.downconv1(enc1))
        enc3 = self.encoder3(self.downconv2(enc2))
        bottleneck = self.bottleneck(self.downconv3(enc3))

        dec3 = self.upconv3(bottleneck)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        dec1 = self.pixel_shuffle_final(self.upconv_final(dec1))
        out = self.sigmoid(self.conv_final(dec1))
        return out

# Test Super-Resolution Model
def test_super_resolution(model_path, low_res_path, base_output_dir, upscale_factor=4):
    model = SRUNet3D(upscale_factor=upscale_factor)

    # Load state_dict and handle DataParallel's "module." prefix
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    if any(key.startswith("module.") for key in state_dict.keys()):
        state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # Move model to available device
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    low_res = imread(low_res_path)
    low_res = torch.tensor(low_res, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device) / 255.0

    with torch.no_grad():
        high_res_pred = model(low_res).squeeze(0).squeeze(0).cpu().numpy()

    # Output directory
    output_dir = os.path.join(base_output_dir, f"x{upscale_factor}_outputs")
    os.makedirs(output_dir, exist_ok=True)

    # Modify filename: scale resolution number and append _NB{n}
    original_filename = os.path.basename(low_res_path)
    new_filename = re.sub(
        r"_(\d+)_", 
        lambda m: f"_{int(m.group(1)) * upscale_factor}_", 
        original_filename
    )
    name, ext = os.path.splitext(new_filename)
    final_filename = f"{name}{ext}"
    save_path = os.path.join(output_dir, final_filename)

    # Save the output
    imwrite(save_path, (high_res_pred * 255).astype(np.uint8))
    print(f"High-resolution output saved to {save_path}")

# Example Usage
if __name__ == "__main__":
    upscale_factor = 2
    model_path = os.path.join(PROJECT_ROOT, "checkpoints", "SRUNet3D", f"model_x{upscale_factor}", f"sr_model_x{upscale_factor}.pth")
    input_dir = os.path.join(PROJECT_ROOT, "data", "Testing_Data", "100")
    output_base_dir = os.path.join(PROJECT_ROOT, "data", "SR_Data", "SRUNet3D")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    tif_files = sorted(glob(os.path.join(input_dir, "*.tif")))
    if not tif_files:
        raise RuntimeError(f"No .tif files found in {input_dir}")

    for tif_path in tif_files:
        test_super_resolution(model_path, tif_path, output_base_dir, upscale_factor)