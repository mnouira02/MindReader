MindReader: Reconstructing Images from Brainwaves using Stable Diffusion

MindReader is an end-to-end deep learning pipeline that decodes visual perception from non-invasive EEG signals. By training a custom ControlNet adapter on 1.1 million EEG trials from the THINGS-EEG dataset, this project successfully reconstructs the "gist"—shapes, colors, and textures—of what a person is seeing, directly from their brainwaves.

🧠 How it Works

The human skull acts as a low-pass filter, blurring neuronal activity into noisy surface signals. To overcome this, MindReader employs a multi-stage approach:

Contrastive Learning (CLIP): We align noisy EEG signals with the semantic latent space of CLIP (Contrastive Language-Image Pre-training). This teaches the model to associate specific brainwave patterns with high-level visual concepts (e.g., "An image of a dog").

Generative Reconstruction (ControlNet): We train a custom "Brain Adapter" that injects these aligned EEG embeddings directly into Stable Diffusion v1.5. This effectively replaces the text prompt with a "brain prompt," steering the image generation process to match the visual features encoded in the user's mind.

Massive Scale: Training was performed on the THINGS-EEG dataset, comprising over 1,854 unique visual concepts across 50 subjects, totaling ~1.1 million single-trial EEG recordings.

🚀 Results

Zero-Shot Retrieval Accuracy: 40.0% (vs 0.8% random chance). The model can correctly identify the exact image seen by a subject from a random lineup of 128 options with high accuracy.

Generative Fidelity: Successfully reconstructs dominant colors, object categories (e.g., food, animals), and geometric shapes (e.g., round vs. linear objects) from single-trial EEG data.

🛠️ Tech Stack

Core Frameworks: PyTorch, HuggingFace Diffusers, Transformers

Data Processing: MNE-Python, Pandas, NumPy

Architecture: 1D CNN Encoder + Multi-Layer Perceptron (MLP) Projector + Frozen Stable Diffusion v1.5

Optimization: Trained on a single NVIDIA RTX 2080 (8GB VRAM) using Gradient Accumulation, Mixed Precision (FP16), and Gradient Checkpointing.

📂 Project Structure

MindReader/
├── assets/                 # Demo images and GIFs
├── data/                   # Raw and preprocessed data (Not included in repo)
├── src/                    # Source code
│   ├── train_contrastive.py        # Stage 1: Train EEG-to-CLIP alignment
│   ├── train_adapter.py            # Stage 2: Train Brain-to-Image Adapter
│   ├── generate_thoughts.py        # Inference: Generate images from brainwaves
│   ├── evaluate_images.py          # Metrics: Calculate CLIP similarity scores
│   └── utils.py                    # Shared model definitions and helpers
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation


⚡ Quick Start

1. Installation

git clone [https://github.com/yourusername/MindReader.git](https://github.com/yourusername/MindReader.git)
cd MindReader
pip install -r requirements.txt


2. Data Setup

This project uses the THINGS-EEG dataset. Due to size (1.5 TB raw), you must download it manually or use the provided helper scripts.

EEG Data: Download from OpenNeuro (ds003825).

Image Data: Download images_THINGS.zip from OSF (jum2f).

Place data in the data/ directory as follows:

data/
  raw_data/       # BIDS format EEG
  things_images/  # Unzipped image folders


3. Preprocessing

Convert raw BIDS data to optimized NumPy arrays:

python src/convert_data.py


4. Running Inference (Pre-trained)

If you have downloaded the pre-trained weights (link coming soon), run:

python src/generate_thoughts.py --model_path checkpoints/mind_reader_adapter.pth


📜 Citation

If you use this code, please cite the original THINGS-EEG paper:

Grootswagers, T., Zhou, I., Robinson, A.K. et al. Human EEG recordings for 1,854 concepts presented in rapid serial visual presentation streams. Sci Data 9, 3 (2022).

📄 License

MIT License. Free for research and educational use.