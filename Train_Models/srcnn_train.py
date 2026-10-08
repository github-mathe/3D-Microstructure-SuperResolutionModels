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



class ResidualBlock(nn.Module):
    def __init__(self, num_features, dropout_prob=0.1):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv3d(num_features, num_features, kernel_size=3, padding=1),
            nn.BatchNorm3d(num_features),
            nn.PReLU(),
            nn.Dropout3d(p=dropout_prob),
            nn.Conv3d(num_features, num_features, kernel_size=3, padding=1),
            nn.BatchNorm3d(num_features),
        )

    def forward(self, x):
        return x + self.block(x)


# Denoising Layer
class DenoisingBlock(nn.Module):
    def __init__(self, num_features, num_layers=3):
        super(DenoisingBlock, self).__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(
                nn.Conv3d(num_features, num_features, kernel_size=3, padding=1)
            )
            layers.append(nn.PReLU())
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)



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

# Training Function
def train_super_resolution(
    low_res_dir, high_res_dir, epochs, batch_size, learning_rate, patience,
    train_samples_ratio, max_samples, upscale_factor, save_dir, augment, crop_size, num_crops,
    resume_checkpoint=None  # Path to checkpoint file
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
    
    # Print dataset sizes at the end
    print(f"\n✅ Data Loaded Successfully!")
    print(f"📂 Training samples loaded: {len(train_dataset)}")
    print(f"📂 Validation samples loaded: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False, num_workers=8, pin_memory=True)

    model = SRCNN3D(upscale_factor).to(device)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    # Resume training if checkpoint exists
    start_epoch = 0
    best_val_loss = float('inf')
    patience_counter = 0
    
    train_losses, val_losses = [], []
    train_psnr, val_psnr = [], []
    train_ssim, val_ssim = [], []

    # 🔥 Check if checkpoint exists
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        print(f"Resuming training from checkpoint: {resume_checkpoint}")
        checkpoint = torch.load(resume_checkpoint, map_location=device)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        start_epoch = checkpoint.get('epoch', 0) + 1  # ✅ Start from next epoch
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        patience_counter = checkpoint.get('patience_counter', 0)

        print(f"Loaded checkpoint from epoch {start_epoch}")

        # 🔥 Load existing training logs
        if os.path.exists(log_file):
            # 🔥 Load existing training logs but ignore metadata lines
            with open(log_file, 'r') as f:
                lines = f.readlines()

            for line in lines:
                if line.startswith("#") or line.startswith("Epoch"):  # ✅ Ignore metadata and headers
                    continue

                values = line.strip().split("\t")
                
                if len(values) < 7:  # ✅ Prevent "index out of range" error
                    continue  

                train_losses.append(float(values[1]))
                val_losses.append(float(values[2]))
                train_psnr.append(float(values[3]))
                val_psnr.append(float(values[4]))
                train_ssim.append(float(values[5]))
                val_ssim.append(float(values[6]))

    else:
        # 🆕 No checkpoint: Create new log file with headers
        print("No checkpoint found, starting new training session.")
        start_epoch = 0  # Ensure new training starts from zero

        with open(log_file, "w") as log:
            log.write(f"# Training Samples: {len(train_dataset)}\n")  # ✅ Save training sample count (once)
            log.write(f"# Validation Samples: {len(val_dataset)}\n")
            log.write("Epoch\tTrain Loss\tVal Loss\tTrain PSNR\tVal PSNR\tTrain SSIM\tVal SSIM\n")
            
    # 🔥 Ensure loop starts from last saved epoch
    for epoch in range(start_epoch, epochs):  # ✅ FIXED: Start from correct epoch
        if epoch == 0:
            start_time = time.time()

        model.train()
        train_loss, train_psnr_epoch, train_ssim_epoch = 0, 0, 0

        for low_res, high_res in train_loader:
            low_res, high_res = low_res.to(device), high_res.to(device)

            outputs = model(low_res)
            loss = criterion(outputs, high_res)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            train_psnr_epoch += compute_psnr_torch(outputs, high_res).item() * low_res.size(0)
            train_ssim_epoch += ssim_metric(outputs, high_res).item() * low_res.size(0)

        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        train_psnr.append(train_psnr_epoch / len(train_loader.dataset))
        train_ssim.append(train_ssim_epoch / len(train_loader.dataset))

        model.eval()
        val_loss, val_psnr_epoch, val_ssim_epoch = 0, 0, 0
        with torch.no_grad():
            for low_res, high_res in val_loader:
                low_res, high_res = low_res.to(device), high_res.to(device)

                outputs = model(low_res)
                loss = criterion(outputs, high_res)
                val_loss += loss.item()

                val_psnr_epoch += compute_psnr_torch(outputs, high_res).item() * low_res.size(0)
                val_ssim_epoch += ssim_metric(outputs, high_res).item() * low_res.size(0)

        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        val_psnr.append(val_psnr_epoch / len(val_loader.dataset))
        val_ssim.append(val_ssim_epoch / len(val_loader.dataset))
        
        if epoch == 0:
            elapsed_time = time.time() - start_time
            print(f"⏱️ First epoch time: {elapsed_time:.2f} seconds")
            with open(log_file, "a") as log:
                log.write(f"# First Epoch Time (seconds): {elapsed_time:.2f}\n")

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_path = os.path.join(save_dir, f"sr_model_x{upscale_factor}.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"✅ model saved at epoch {epoch + 1} with validation loss: {val_loss:.6f}")
        else:
            patience_counter += 1

        # Save checkpoint after every epoch
        torch.save({
            'epoch': epoch,  # ✅ Ensure correct epoch is saved
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'patience_counter': patience_counter,
        }, checkpoint_path)
        print(f"Checkpoint saved at epoch {epoch + 1}")
        
        # Save the model every 100 epochs
        if (epoch + 1) % 100 == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, f"model_epoch_{epoch + 1}.pth"))
            print(f"Model saved at epoch {epoch + 1}")

        # Append to training log
        with open(log_file, "a") as log:
            log.write(f"{epoch + 1}\t{train_loss:.8f}\t{val_loss:.8f}\t"
                      f"{train_psnr[-1]:.3f}\t{val_psnr[-1]:.3f}\t"
                      f"{train_ssim[-1]:.3f}\t{val_ssim[-1]:.3f}\n")

        # Early stopping
        if patience_counter >= patience:
            print(f"🚨 Early stopping triggered at epoch {epoch + 1}.")
            break

        print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.8f}, Val Loss: {val_loss:.8f}")

        # Plot metrics
        plt.figure()
        plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss")
        plt.plot(range(1, len(val_losses) + 1), val_losses, label="Val Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Training and Validation Loss")
        plt.grid()
        plt.savefig(os.path.join(save_dir, "loss_curve.png"))
        plt.close()

        plt.figure()
        plt.plot(range(1, len(train_psnr) + 1), train_psnr, label="Train PSNR")
        plt.plot(range(1, len(val_psnr) + 1), val_psnr, label="Val PSNR")
        plt.xlabel("Epochs")
        plt.ylabel("PSNR")
        plt.legend()
        plt.title("Training and Validation PSNR")
        plt.grid()
        plt.savefig(os.path.join(save_dir, "psnr_curve.png"))
        plt.close()

        plt.figure()
        plt.plot(range(1, len(train_ssim) + 1), train_ssim, label="Train SSIM")
        plt.plot(range(1, len(val_ssim) + 1), val_ssim, label="Val SSIM")
        plt.xlabel("Epochs")
        plt.ylabel("SSIM")
        plt.legend()
        plt.title("Training and Validation SSIM")
        plt.grid()
        plt.savefig(os.path.join(save_dir, "ssim_curve.png"))
        plt.close()

    print("Training complete.")


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
    batch_size = 64
    learning_rate = 1e-4
    patience = 20
    train_sample_ratio = 0.8
    max_samples = None
    save_dir = os.path.join(PROJECT_ROOT, "checkpoints", "SRCNN3D", f"model_x{upscale_factor}")
    augment = False
    crop_size = (24, 24, 24)
    num_crops = 30
    # resume_checkpoint = "model_x2/checkpoint.pth"
    resume_checkpoint = None

    train_super_resolution(
    low_res_dir, high_res_dir,
    epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, patience=patience,
    train_samples_ratio=train_sample_ratio, max_samples=max_samples, upscale_factor=upscale_factor,
    save_dir=save_dir, augment=augment, crop_size=crop_size, num_crops=num_crops,
    resume_checkpoint=resume_checkpoint  # Resume training
    )

