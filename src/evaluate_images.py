import torch
import numpy as np
from diffusers import StableDiffusionPipeline
from transformers import CLIPProcessor, CLIPModel
import torch.nn as nn
import os
import random
import glob
from PIL import Image
from tqdm import tqdm

# --- CONFIGURATION ---
DATA_DIR = "things_eeg_data"
IMAGE_DIR = "things_images"
CHECKPOINT_DIR = "mind_reader_adapter"
NUM_SAMPLES = 50  # Test on 50 images to get a good average

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Running Image-to-Image Evaluation on: {device.upper()}")

# ==========================================
# 1. BRAIN ENCODER (Must match training)
# ==========================================
class BrainEncoder(nn.Module):
    def __init__(self, num_channels=63, time_points=91):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(num_channels, 128, 32, padding=16), nn.GroupNorm(8, 128), nn.GELU(), nn.MaxPool1d(2),
            nn.Conv1d(128, 256, 16, padding=8), nn.GroupNorm(16, 256), nn.GELU(), nn.MaxPool1d(2),
            nn.Conv1d(256, 512, 8, padding=4), nn.GroupNorm(32, 512), nn.GELU(), nn.MaxPool1d(2),
            nn.Flatten()
        )
        with torch.no_grad():
            dummy = torch.zeros(1, num_channels, time_points)
            flat_out = self.features(dummy).shape[1]
        self.adapter = nn.Sequential(
            nn.Linear(flat_out, 4096), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(4096, 768) 
        )
    def forward(self, x):
        feat = self.features(x)
        return self.adapter(feat).unsqueeze(1).repeat(1, 77, 1)

# ==========================================
# 2. MAIN EVALUATION LOOP
# ==========================================
def evaluate_reconstruction():
    # 1. Load Brain Adapter
    checkpoints = glob.glob(f"{CHECKPOINT_DIR}/*.pth")
    if not checkpoints: return
    latest_model = max(checkpoints, key=os.path.getctime)
    
    # Detect shape
    sample_files = glob.glob(os.path.join(DATA_DIR, "arr_0_sub-*.npy"))
    sample_data = np.load(sample_files[0], mmap_mode='r')
    
    adapter = BrainEncoder(num_channels=sample_data.shape[1], time_points=sample_data.shape[2]).to(device)
    adapter.load_state_dict(torch.load(latest_model), strict=False)
    adapter.eval()

    # 2. Load Stable Diffusion (Generation)
    print("⏳ Loading Stable Diffusion...")
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, variant="fp16")
    pipe.enable_model_cpu_offload() # Save VRAM
    pipe.safety_checker = None

    # 3. Load CLIP (Scoring)
    print("⏳ Loading CLIP Vision Model...")
    # We use the Vision Model this time, not the Text Model
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # 4. Load Data
    print("📡 Loading Test Data...")
    target_sub_x = os.path.join(DATA_DIR, "arr_0_sub-01.npy")
    target_sub_y = os.path.join(DATA_DIR, "arr_1_sub-01.npy")
    X_test = np.load(target_sub_x).astype(np.float32)
    y_test = np.load(target_sub_y, allow_pickle=True)
    
    # Normalize
    X_test = (X_test - np.mean(X_test, axis=2, keepdims=True)) / (np.std(X_test, axis=2, keepdims=True) + 1e-6)
    
    # Filter valid
    valid_indices = [i for i, label in enumerate(y_test) if label not in ['-1', 'n/a', 'nan']]
    
    # Randomly select samples
    indices = random.sample(valid_indices, NUM_SAMPLES)
    
    print(f"\n📊 Evaluating {NUM_SAMPLES} samples...")
    scores = []
    
    for i, idx in enumerate(tqdm(indices)):
        eeg_sample = torch.FloatTensor(X_test[idx]).unsqueeze(0).to(device)
        label = y_test[idx]
        
        # A. Generate Image from Brain
        with torch.no_grad():
            brain_prompt = adapter(eeg_sample)
        
        # Note: We generate small thumbnails (256x256) to speed up evaluation
        image_gen = pipe(prompt_embeds=brain_prompt, num_inference_steps=20, height=256, width=256).images[0]
        
        # B. Load Real Image
        concept_path = os.path.join(IMAGE_DIR, label)
        real_image = None
        if os.path.exists(concept_path):
            images = os.listdir(concept_path)
            if images:
                real_image = Image.open(os.path.join(concept_path, images[0])).convert("RGB")
        
        if real_image is None: continue # Skip if no ground truth

        # C. Compute CLIP Similarity (Image vs Image)
        with torch.no_grad():
            # Process both images
            inputs = clip_processor(images=[real_image, image_gen], return_tensors="pt").to(device)
            
            # Get image embeddings
            image_features = clip_model.get_image_features(**inputs)
            
            # Normalize
            image_features = image_features / image_features.norm(dim=1, keepdim=True)
            
            # Calculate Cosine Similarity
            # Row 0 is Real, Row 1 is Gen. We want their dot product.
            similarity = (image_features[0] @ image_features[1].T).item()
            scores.append(similarity)

    avg_score = np.mean(scores)
    print("\n" + "="*40)
    print(f"🏆 Average CLIP Image Similarity: {avg_score:.4f}")
    print("="*40)
    print("Interpretation:")
    print("   < 0.60: Random / Noise")
    print("   0.60 - 0.70: Vague conceptual match")
    print("   > 0.70: Strong visual/semantic match")
    print("   > 0.85: Near-perfect reconstruction")

if __name__ == "__main__":
    evaluate_reconstruction()