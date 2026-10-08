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

    
class Discriminator3D(nn.Module):
    def __init__(self):
        super(Discriminator3D, self).__init__()
        self.net = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            
            nn.Conv3d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.2),
            
            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2),
            
            nn.Conv3d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2),
            
            nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2),
            
            nn.Conv3d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2),
            
            nn.Conv3d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(512),
            nn.LeakyReLU(0.2),
            
            nn.Conv3d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(512),
            nn.LeakyReLU(0.2),
            
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x).view(-1)  # Ensures output shape is (batch_size,)

# PSNR/MSE helpers supporting both CPU and GPU
def compute_mse_torch(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    """Compute Mean Squared Error between two tensors."""
    return torch.mean((img1 - img2) ** 2)

def compute_psnr_torch(img1: torch.Tensor, img2: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    """Compute PSNR (dB) between two tensors with values in [0,max_val]."""
    mse = compute_mse_torch(img1, img2)
    if mse == 0:
        return torch.tensor(float('inf'), device=img1.device)
    return 20 * torch.log10(max_val / torch.sqrt(mse))

class CPUGatherDataParallel(nn.DataParallel):
    def gather(self, outputs, output_device):
        # move each chunk back to CPU, concat there, then send to output_device
        cpu_cat = torch.cat([out.cpu() for out in outputs], dim=0)
        return cpu_cat.to(f"cuda:{output_device}")

def train_super_resolution(
    low_res_dir, high_res_dir, epochs, batch_size, learning_rate, patience,
    train_samples_ratio, max_samples, upscale_factor, save_dir, augment,
    crop_size, num_crops, num_residual_block,
    resume_checkpoint=None
):
    os.makedirs(save_dir, exist_ok=True)

    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    log_file = os.path.join(save_dir, "training_log.txt")
    checkpoint_path = os.path.join(save_dir, "checkpoint.pth")

    dataset = SuperResolutionDataset(low_res_dir, high_res_dir, crop_size, 
                                     upscale_factor, num_crops, augment)
    
    if max_samples is not None:
        dataset = Subset(dataset, range(min(len(dataset), max_samples)))

    train_size = int(train_samples_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"\n✅ Data Loaded Successfully!")
    print(f"📂 Training samples loaded: {len(train_dataset)}")
    print(f"📂 Validation samples loaded: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=16, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False, num_workers=16, pin_memory=True)
    

    generator_model = Generator3D(upscale_factor, num_residual_block).to(device)
    discriminator_model = Discriminator3D().to(device)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        generator_model = nn.DataParallel(generator_model)
        discriminator_model = nn.DataParallel(discriminator_model)

    criterion = nn.MSELoss()
    adversarial_loss = nn.BCEWithLogitsLoss()
    optimizer_G = optim.AdamW(generator_model.parameters(), lr=learning_rate, weight_decay=1e-5)
    optimizer_D = optim.AdamW(discriminator_model.parameters(), lr=learning_rate, weight_decay=1e-5)

    start_epoch = 0
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses, val_losses, disc_losses = [], [], []
    train_psnr, val_psnr = [], []
    train_ssim, val_ssim = [], []

    if resume_checkpoint and os.path.exists(resume_checkpoint):
        print(f"Resuming training from checkpoint: {resume_checkpoint}")
        checkpoint = torch.load(resume_checkpoint, map_location=device)
        generator_model.load_state_dict(checkpoint['generator_state_dict'])
        discriminator_model.load_state_dict(checkpoint['discriminator_state_dict'])
        optimizer_G.load_state_dict(checkpoint['optimizer_G'])
        optimizer_D.load_state_dict(checkpoint['optimizer_D'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        patience_counter = checkpoint.get('patience_counter', 0)
        print(f"Checkpoint loaded. Resuming from epoch {start_epoch}")

        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                if line.startswith("#") or line.startswith("Epoch"):
                    continue
                values = line.strip().split("\t")
                if len(values) < 8:
                    continue  
                train_losses.append(float(values[1]))
                val_losses.append(float(values[2]))
                disc_losses.append(float(values[3]))
                train_psnr.append(float(values[4]))
                val_psnr.append(float(values[5]))
                train_ssim.append(float(values[6]))
                val_ssim.append(float(values[7]))
    else:
        with open(log_file, "w") as log:
            log.write(f"# Training Samples: {len(train_dataset)}\n")
            log.write(f"# Validation Samples: {len(val_dataset)}\n")
            log.write("Epoch\tTrain Loss\tVal Loss\tDisc Loss\tTrain PSNR\tVal PSNR\tTrain SSIM\tVal SSIM\n")



    for epoch in range(start_epoch, epochs):
        if epoch == 0:
            start_time = time.time()

        generator_model.train()
        discriminator_model.train()
        train_loss, gen_loss_epoch, disc_loss_epoch = 0, 0, 0
        train_psnr_epoch, train_ssim_epoch = 0, 0

        for low_res, high_res in train_loader:
            low_res, high_res = low_res.to(device), high_res.to(device)
            
            real_labels = torch.ones((low_res.size(0),), device=device)
            fake_labels = torch.zeros((low_res.size(0),), device=device)
            
            fake_high_res = generator_model(low_res)
            real_output = discriminator_model(high_res)
            fake_output = discriminator_model(fake_high_res.detach())
            
            loss_D = adversarial_loss(real_output, real_labels) + adversarial_loss(fake_output, fake_labels)
            optimizer_D.zero_grad()
            loss_D.backward()
            optimizer_D.step()
            disc_loss_epoch += loss_D.item()
            
            fake_output = discriminator_model(fake_high_res)
            loss_G = criterion(fake_high_res, high_res) + 0.001 * adversarial_loss(fake_output, real_labels)
            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()
            
            gen_loss_epoch += loss_G.item()
            train_psnr_epoch += compute_psnr_torch(fake_high_res, high_res).item() * low_res.size(0)
            train_ssim_epoch += ssim_metric(fake_high_res, high_res).item() * low_res.size(0)


        train_losses.append(gen_loss_epoch / len(train_loader))
        disc_losses.append(disc_loss_epoch / len(train_loader))
        train_psnr.append(train_psnr_epoch / len(train_loader.dataset))
        train_ssim.append(train_ssim_epoch / len(train_loader.dataset))

        # Validation
        generator_model.eval()
        val_loss, val_psnr_epoch, val_ssim_epoch = 0, 0, 0
        with torch.no_grad():
            for low_res, high_res in val_loader:
                low_res, high_res = low_res.to(device), high_res.to(device)
                
                outputs = generator_model(low_res)
                val_loss += criterion(outputs, high_res).item()
                val_psnr_epoch += compute_psnr_torch(outputs, high_res).item() * low_res.size(0)
                val_ssim_epoch += ssim_metric(outputs, high_res).item() * low_res.size(0)

        val_losses.append(val_loss / len(val_loader))
        val_psnr.append(val_psnr_epoch / len(val_loader.dataset))
        val_ssim.append(val_ssim_epoch / len(val_loader.dataset))
        
        if epoch == 0:
            elapsed_time = time.time() - start_time
            print(f"⏱️ First epoch time: {elapsed_time:.2f} seconds")
            with open(log_file, "a") as log:
                log.write(f"# First Epoch Time (seconds): {elapsed_time:.2f}\n")

        # Save Best Model
        if val_losses[-1] < best_val_loss:
            best_val_loss = val_losses[-1]
            patience_counter = 0
            torch.save(generator_model.state_dict(), os.path.join(save_dir, f"sr_generator_x{upscale_factor}.pth"))
            torch.save(discriminator_model.state_dict(), os.path.join(save_dir, f"sr_discriminator_x{upscale_factor}.pth"))
            print(f"✅ model saved at epoch {epoch + 1}")

        else:
            patience_counter += 1

        # Save Model Every 100 Epochs
        if (epoch + 1) % 50 == 0:
            torch.save(generator_model.state_dict(), os.path.join(save_dir, f"generator_epoch_{epoch+1}.pth"))
            torch.save(discriminator_model.state_dict(), os.path.join(save_dir, f"discriminator_epoch_{epoch+1}.pth"))
            print(f"✅ Model saved at epoch {epoch + 1}")

        # Save Checkpoint Every Epoch
        torch.save({
            'epoch': epoch,
            'generator_state_dict': generator_model.state_dict(),
            'discriminator_state_dict': discriminator_model.state_dict(),
            'optimizer_G': optimizer_G.state_dict(),
            'optimizer_D': optimizer_D.state_dict(),
            'best_val_loss': best_val_loss,
            'patience_counter': patience_counter,
        }, checkpoint_path)
        print(f"✅ Checkpoint saved at epoch {epoch + 1}")

        # Early Stopping
        if patience_counter >= patience:
            print(f"🚨 Early stopping triggered at epoch {epoch + 1}.")
            break

        # Save Training Log
        with open(log_file, "a") as log:
            log.write(f"{epoch + 1}\t{train_losses[-1]:.8f}\t{val_losses[-1]:.8f}\t{disc_losses[-1]:.8f}\t"
                      f"{train_psnr[-1]:.3f}\t{val_psnr[-1]:.3f}\t{train_ssim[-1]:.3f}\t{val_ssim[-1]:.3f}\n")


        print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {train_losses[-1]:.8f}, Val Loss: {val_losses[-1]:.8f}")

        # Save loss graph
        plt.figure()
        plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss")
        plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Training and Validation Loss")
        plt.grid()
        plt.savefig(os.path.join(save_dir, "loss_curve.png"))
        plt.close()

        plt.figure()
        plt.plot(range(1, len(disc_losses) + 1), disc_losses, label="Discriminator Loss")
        plt.xlabel("Epochs")
        plt.ylabel("D_Loss")
        plt.legend()
        plt.title("Discriminator Loss")
        plt.grid()
        plt.savefig(os.path.join(save_dir, "d_loss_curve.png"))
        plt.close()

        plt.figure()
        plt.plot(range(1, len(train_psnr) + 1), train_psnr, label="Train PSNR")
        plt.plot(range(1, len(val_psnr) + 1), val_psnr, label="Validation PSNR")
        plt.xlabel("Epochs")
        plt.ylabel("PSNR")
        plt.legend()
        plt.title("Training and Validation PSNR")
        plt.grid()
        plt.savefig(os.path.join(save_dir, "psnr_curve.png"))
        plt.close()

        plt.figure()
        plt.plot(range(1, len(train_ssim) + 1), train_ssim, label="Train SSIM")
        plt.plot(range(1, len(val_ssim) + 1), val_ssim, label="Validation SSIM")
        plt.xlabel("Epochs")
        plt.ylabel("SSIM")
        plt.legend()
        plt.title("Training and Validation SSIM")
        plt.grid()
        plt.savefig(os.path.join(save_dir, "ssim_curve.png"))
        plt.close()

    print("🔥 Training Complete!")


# Example Usage
if __name__ == "__main__":
    # Repository-relative training data roots
    low_res_base_dir = os.path.join(PROJECT_ROOT, "data", "Training_Data")
    high_res_base_dir = low_res_base_dir

    # Directory labels: low-res always 100; high-res = 100 * upscale_factor
    LOW_LABEL = 100

    # Set desired upscale factor
    upscale_factor = 2  # or 4, 8

    # Validate upscale factor
    if upscale_factor not in (2, 4, 8):
        raise ValueError("Unsupported upscale factor. Use 2, 4, or 8.")

    # Construct paths
    low_res_dir = os.path.join(low_res_base_dir, str(LOW_LABEL))
    high_res_dir = os.path.join(high_res_base_dir, str(LOW_LABEL * upscale_factor))

    print(f"Low-res dir:  {low_res_dir}")
    print(f"High-res dir: {high_res_dir}")
    
    epochs = 500
    batch_size = 32
    learning_rate = 1e-4
    patience = 100
    train_sample_ratio = 0.8
    max_samples = None
    augment = False
    crop_size = (24, 24, 24)
    num_crops = 30
    # resume_checkpoint = os.path.join("checkpoints", "SRGAN3D", "model_x2", "checkpoint.pth")
    resume_checkpoint = None
    
    residual_block_list = [16]

    for num_residual_block in residual_block_list:
        save_dir = os.path.join(PROJECT_ROOT, "checkpoints", "SRGAN3D", f"model_x{upscale_factor}")

        print(f"\n🚀 Starting training for Residual Blocks = {num_residual_block}")
        
        train_super_resolution(
            low_res_dir=low_res_dir,
            high_res_dir=high_res_dir,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            patience=patience,
            train_samples_ratio=train_sample_ratio,
            max_samples=max_samples,
            upscale_factor=upscale_factor,
            save_dir=save_dir,
            augment=augment,
            crop_size=crop_size,
            num_crops=num_crops,  # You may adjust this
            num_residual_block=num_residual_block,
            resume_checkpoint=resume_checkpoint
        )

