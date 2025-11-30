import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from diffusers import StableDiffusionPipeline, AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from PIL import Image
from torchvision import transforms
import numpy as np
import glob
import random
import sys

# Import the shared BrainEncoder model
# Ensure you run this script from the src folder, or adjust sys.path if running from root
try:
    from utils import BrainEncoder
except ImportError:
    # Fallback if running from root directory
    sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
    from src.utils import BrainEncoder

# --- CONFIGURATION (ULTRA LOW VRAM) ---
DATA_DIR = "things_eeg_data"
IMAGE_DIR = "things_images"
OUTPUT_DIR = "mind_reader_adapter"

BATCH_SIZE = 1          
GRAD_ACCUM_STEPS = 8    # Increased accumulation for stability
EPOCHS = 10      
LR = 1e-5        
SAVE_EVERY_STEPS = 5000 

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Running ULTRA LOW VRAM Adapter Training on: {device.upper()}")

# ==========================================
# 1. DATASET
# ==========================================
class EEGImageDataset(Dataset):
    def __init__(self, eeg_dir, img_dir):
        self.eeg_dir = eeg_dir
        self.img_dir = img_dir
        self.samples = []
        
        self.transform = transforms.Compose([
            transforms.Resize(512),
            transforms.CenterCrop(512),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
        print("   Indexing dataset pairs...")
        x_files = sorted(glob.glob(os.path.join(eeg_dir, "arr_0_sub-*.npy")))
        
        for x_f in x_files:
            try:
                y_f = x_f.replace("arr_0", "arr_1")
                if not os.path.exists(y_f): continue
                labels = np.load(y_f, allow_pickle=True)
                for i, label in enumerate(labels):
                    if label in ['-1', 'n/a', 'nan']: continue
                    concept_path = os.path.join(self.img_dir, label)
                    if os.path.isdir(concept_path):
                        self.samples.append((x_f, i, label))
            except: pass
            
        print(f"   ✅ Found {len(self.samples)} valid pairs.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x_path, x_idx, label = self.samples[idx]
        X_full = np.load(x_path, mmap_mode='r')
        eeg_raw = X_full[x_idx].copy()
        
        eeg_mean = np.mean(eeg_raw)
        eeg_std = np.std(eeg_raw) + 1e-6
        eeg_tensor = torch.tensor((eeg_raw - eeg_mean) / eeg_std).float()
        
        concept_path = os.path.join(self.img_dir, label)
        try:
            images = os.listdir(concept_path)
            if images:
                img_name = random.choice(images)
                image = Image.open(os.path.join(concept_path, img_name)).convert("RGB")
                pixel_values = self.transform(image)
            else:
                pixel_values = torch.zeros(3, 512, 512)
        except:
            pixel_values = torch.zeros(3, 512, 512)

        return eeg_tensor, pixel_values

# ==========================================
# 2. TRAINING LOOP
# ==========================================
def train_adapter():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("⏳ Loading Stable Diffusion components (FP16)...")
    noise_scheduler = DDPMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler")
    vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae", torch_dtype=torch.float16).to(device)
    unet = UNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="unet", torch_dtype=torch.float16).to(device)
    
    # Enable Gradient Checkpointing for VRAM savings
    unet.enable_gradient_checkpointing()
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    
    # Dynamic Data Shape Check
    sample_files = glob.glob(os.path.join(DATA_DIR, "arr_0_sub-*.npy"))
    if not sample_files:
        print("❌ No data found.")
        return
    sample_data = np.load(sample_files[0], mmap_mode='r')
    real_channels = sample_data.shape[1]
    real_time_points = sample_data.shape[2]
    print(f"   Detected Data Shape: Channels={real_channels}, Time={real_time_points}")

    # Initialize Adapter (Imported from utils)
    adapter = BrainEncoder(num_channels=real_channels, time_points=real_time_points).to(device)
    adapter.train()
    
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=LR)
    scaler = torch.cuda.amp.GradScaler()
    
    dataset = EEGImageDataset(DATA_DIR, IMAGE_DIR)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    print(f"👟 Training Adapter ({EPOCHS} Epochs)...")
    
    for epoch in range(EPOCHS):
        adapter.train()
        total_loss = 0
        optimizer.zero_grad() 
        
        for step, (eeg, images) in enumerate(dataloader):
            eeg = eeg.to(device)
            images = images.to(device, dtype=torch.float16)
            
            # FP16 Autocast Context
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    latents = vae.encode(images).latent_dist.sample()
                    latents = latents * 0.18215
                    
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                # Forward Pass
                encoder_hidden_states = adapter(eeg)
                # Cast to FP16 for UNet compatibility
                encoder_hidden_states_fp16 = encoder_hidden_states.to(dtype=torch.float16)
                
                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states_fp16).sample
                
                loss = torch.nn.functional.mse_loss(noise_pred, noise)
                loss = loss / GRAD_ACCUM_STEPS
            
            # Backward Pass
            scaler.scale(loss).backward()
            
            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            total_loss += loss.item() * GRAD_ACCUM_STEPS
            
            # Logging & Checkpointing
            if step % 200 == 0:
                print(f"   Ep {epoch+1} [{step}/{len(dataloader)}] Loss: {loss.item() * GRAD_ACCUM_STEPS:.4f}")
            
            # --- SAFETY SAVE EVERY 5000 STEPS ---
            if step > 0 and step % SAVE_EVERY_STEPS == 0:
                checkpoint_path = f"{OUTPUT_DIR}/adapter_checkpoint.pth"
                print(f"   💾 Safety Save: {checkpoint_path}")
                torch.save(adapter.state_dict(), checkpoint_path)
                
        avg_loss = total_loss / len(dataloader)
        print(f"💾 Epoch {epoch+1} Complete. Avg Loss: {avg_loss:.4f}")
        torch.save(adapter.state_dict(), f"{OUTPUT_DIR}/adapter_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    train_adapter()