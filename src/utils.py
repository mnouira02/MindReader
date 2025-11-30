import torch
import torch.nn as nn
import numpy as np

# ==========================================
# MODEL DEFINITION
# ==========================================
class BrainEncoder(nn.Module):
    """
    The core neural network for MindReader.
    Input: EEG Signal (Batch, Channels, Time)
    Output: Latent Embedding (Batch, 1, 77, 768) for Stable Diffusion
    """
    def __init__(self, num_channels=63, time_points=91):
        super().__init__()
        
        # Feature Extractor (1D CNN)
        # Designed to capture temporal features from EEG waves
        self.features = nn.Sequential(
            nn.Conv1d(num_channels, 128, kernel_size=32, padding=16), 
            nn.GroupNorm(8, 128), nn.GELU(), nn.MaxPool1d(2),
            
            nn.Conv1d(128, 256, kernel_size=16, padding=8), 
            nn.GroupNorm(16, 256), nn.GELU(), nn.MaxPool1d(2),
            
            nn.Conv1d(256, 512, kernel_size=8, padding=4), 
            nn.GroupNorm(32, 512), nn.GELU(), nn.MaxPool1d(2),
            
            nn.Flatten()
        )
        
        # Dynamically calculate flattened size
        with torch.no_grad():
            dummy = torch.zeros(1, num_channels, time_points)
            flat_out = self.features(dummy).shape[1]

        # Projector Head (The "Adapter")
        # Maps extracted features to the CLIP embedding space (768 dim)
        self.adapter = nn.Sequential(
            nn.Linear(flat_out, 4096), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(4096, 768) 
        )

    def forward(self, x):
        feat = self.features(x)
        embed = self.adapter(feat) 
        # Reshape to match Stable Diffusion's expected text embedding shape
        # (Batch, Sequence_Length, Embedding_Dim) -> (Batch, 77, 768)
        return embed.unsqueeze(1).repeat(1, 77, 1)

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def find_best_match(predicted_vec, concept_map):
    """
    Finds the closest semantic concept (text label) for a given predicted vector.
    Uses Cosine Similarity.
    """
    best_concept = "Unknown"
    best_sim = -1.0
    
    # Normalize input vector
    pred_norm = predicted_vec / (np.linalg.norm(predicted_vec) + 1e-8)
    
    for concept, emb in concept_map.items():
        # Normalize concept vector
        emb_norm = emb / (np.linalg.norm(emb) + 1e-8)
        
        # Calculate Cosine Similarity (Dot product of normalized vectors)
        sim = np.dot(pred_norm, emb_norm)
        
        if sim > best_sim:
            best_sim = sim
            best_concept = concept
            
    return best_concept, best_sim