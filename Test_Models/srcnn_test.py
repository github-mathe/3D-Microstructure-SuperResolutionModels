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
        x = x.view(
            batch_size, channels, r, r, r, depth, height, width
        )
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4)
        return x.reshape(batch_size, channels, depth * r, height * r, width * r)



class SRCNN3D(nn.Module):
    def __init__(self, upscale_factor=2):
        super(SRCNN3D, self).__init__()
        self.upscale_factor = upscale_factor

        # Feature Extraction
        self.conv1 = nn.Conv3d(1, 64, kernel_size=9, padding=4)
        self.prelu1 = nn.PReLU()
        # Non-linear Mapping
        self.conv2 = nn.Conv3d(64, 32, kernel_size=1)
        self.prelu2 = nn.PReLU()
        # Reconstruction
        self.conv3 = nn.Conv3d(32, 32, kernel_size=5, padding=2)
        self.prelu3 = nn.PReLU()
        # Extra conv before all upsampling
        self.conv_pre_up1 = nn.Conv3d(32, 32, kernel_size=3, padding=1)
        self.prelu_pre_up1 = nn.PReLU()

        if self.upscale_factor == 2:
            # single upsampling
            self.up1 = nn.ConvTranspose3d(32, 1, kernel_size=4, stride=2, padding=1)
        elif self.upscale_factor == 4:
            # two upsampling stages
            self.up1 = nn.ConvTranspose3d(32, 16, kernel_size=4, stride=2, padding=1)
            self.prelu_up1 = nn.PReLU()
            self.conv_pre_up2 = nn.Conv3d(16, 16, kernel_size=3, padding=1)
            self.prelu_pre_up2 = nn.PReLU()
            self.up2 = nn.ConvTranspose3d(16, 1, kernel_size=4, stride=2, padding=1)
        else:  # upscale_factor == 8
            # three upsampling stages
            self.up1 = nn.ConvTranspose3d(32, 16, kernel_size=4, stride=2, padding=1)
            self.prelu_up1 = nn.PReLU()
            self.conv_pre_up2 = nn.Conv3d(16, 16, kernel_size=3, padding=1)
            self.prelu_pre_up2 = nn.PReLU()
            self.up2 = nn.ConvTranspose3d(16, 8, kernel_size=4, stride=2, padding=1)
            self.prelu_up2 = nn.PReLU()
            self.conv_pre_up3 = nn.Conv3d(8, 8, kernel_size=3, padding=1)
            self.prelu_pre_up3 = nn.PReLU()
            self.up3 = nn.ConvTranspose3d(8, 1, kernel_size=4, stride=2, padding=1)

        self.final_activation = nn.Sigmoid()

    def forward(self, x):
        x = self.prelu1(self.conv1(x))
        x = self.prelu2(self.conv2(x))
        x = self.prelu3(self.conv3(x))
        # first pre-up conv
        x = self.prelu_pre_up1(self.conv_pre_up1(x))
        if self.upscale_factor == 2:
            x = self.up1(x)
        elif self.upscale_factor == 4:
            x = self.up1(x)
            x = self.prelu_up1(x)
            x = self.prelu_pre_up2(self.conv_pre_up2(x))
            x = self.up2(x)
        else:  # upscale_factor == 8
            x = self.up1(x)
            x = self.prelu_up1(x)
            x = self.prelu_pre_up2(self.conv_pre_up2(x))
            x = self.up2(x)
            x = self.prelu_up2(x)
            x = self.prelu_pre_up3(self.conv_pre_up3(x))
            x = self.up3(x)
        return self.final_activation(x)
    


def test_super_resolution(model_path, low_res_path, base_output_dir, upscale_factor=4):
    # Initialize the model
    model = SRCNN3D(upscale_factor=upscale_factor)

    # Load state_dict and handle DataParallel's "module." prefix
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    if any(key.startswith("module.") for key in state_dict.keys()):
        state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()

    # Move model to available device
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Load and process low-resolution input
    low_res = imread(low_res_path)
    low_res = torch.tensor(low_res, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device) / 255.0

    # Predict high-resolution output
    with torch.no_grad():
        high_res_pred = model(low_res).squeeze(0).squeeze(0).cpu().numpy()

    # Construct output directory
    output_dir = os.path.join(base_output_dir, f"x{upscale_factor}_outputs")
    os.makedirs(output_dir, exist_ok=True)

    # Construct new filename with resolution number replaced
    original_filename = os.path.basename(low_res_path)
    new_filename = re.sub(
        r"_(\d+)_",
        lambda m: f"_{int(m.group(1)) * upscale_factor}_",
        original_filename
    )

    # Final output path
    save_path = os.path.join(output_dir, new_filename)

    # Save the output
    imwrite(save_path, (high_res_pred * 255).astype(np.uint8))
    print(f"High-resolution output saved to {save_path}")

# Example Usage
if __name__ == "__main__":
    upscale_factor = 2
    model_path = os.path.join(PROJECT_ROOT, "checkpoints", "SRCNN3D", f"model_x{upscale_factor}", f"sr_model_x{upscale_factor}.pth")
    input_dir = os.path.join(PROJECT_ROOT, "data", "Testing_Data", "100")
    output_base_dir = os.path.join(PROJECT_ROOT, "data", "SR_Data", "SRCNN3D")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    tif_files = sorted(glob(os.path.join(input_dir, "*.tif")))
    if not tif_files:
        raise RuntimeError(f"No .tif files found in {input_dir}")

    for tif_path in tif_files:
        test_super_resolution(model_path, tif_path, output_base_dir, upscale_factor)