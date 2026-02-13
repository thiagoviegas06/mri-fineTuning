"""
MRI Super-Resolution Training Pipeline

Full Workflow:
1. Preprocessing (preprocessing.py):
   - Load low-field and high-field NIfTI volumes
   - Reorient to canonical (RAS) coordinate system
   - Resample low-field to match high-field grid (physical alignment)
   - Optionally apply rigid registration for anatomy alignment
   - Result: aligned, same-shape LF/HF volumes

2. Data Loading (this file):
   - Extract 96x96 2D patches from each Z-slice
   - Store only patch coordinates (memory efficient)
   - Extract patches on-the-fly during training (lazy loading)

3. Model (model.py):
   - EDSR_X1_Refiner: residual refinement network (2D)
   - Predicts residual (HF - LF), not absolute intensity
   - Fine-tune from pretrained 2x EDSR (backbone transfer learning)
   - Freeze ResBlocks, train only head/tail layers

4. Training (this file):
   - Loss: L1(predicted_residual, true_residual)
   - 80/20 train/val split
   - Validation without gradients
   - No checkpoint saving (add if needed)
"""

import torch 
from torch.utils.data import Dataset, DataLoader, random_split
from preprocessing import pre_process_volume
from model import EDSR_X1_Refiner, load_pretrained_backbone
from huggingface_hub import hf_hub_download
import numpy as np
from torchvision.metrics import structural_similarity_index_measure as ssim
from torchvision.metrics import peak_signal_noise_ratio as psnr


class MRIPatchDataset(Dataset):
    """
    PyTorch Dataset for paired LF/HF 2D MRI patches.
    
    Workflow:
    1. Load aligned low-field and high-field volumes via pre_process_volume()
    2. Pre-compute patch coordinate locations (z, x, y) without loading all patches into RAM
    3. During training, extract patches on-the-fly by indexing into the cached volumes
    
    This keeps memory usage low while allowing efficient batch sampling.
    """
    
    def __init__(self, lf_path, hf_path, patch_size=96, stride=48):
        self.lf_path = lf_path
        self.hf_path = hf_path
        self.patch_size = patch_size
        self.stride = stride
        self.patch_coords = []  # List of (z, x, y) patch coordinate tuples
        
        # Load and preprocess volumes once (align, resample, normalize)
        self.lf_data, self.hf_data = pre_process_volume(lf_path, hf_path)
        
        # Pre-compute patch coordinates from each 2D slice
        for z in range(self.lf_data.shape[2]):
            lf_slice = self.lf_data[:, :, z]
            for x in range(0, lf_slice.shape[0] - patch_size + 1, stride):
                for y in range(0, lf_slice.shape[1] - patch_size + 1, stride):
                    self.patch_coords.append((z, x, y))

    def __len__(self):
        return len(self.patch_coords)

    def __getitem__(self, idx):
        # Extract 2D patch from pre-computed coordinate location
        z, x, y = self.patch_coords[idx]
        lf_patch = self.lf_data[x:x+self.patch_size, y:y+self.patch_size, z]
        hf_patch = self.hf_data[x:x+self.patch_size, y:y+self.patch_size, z]
        
        # Convert to tensor and add channel dimension: (1, patch_size, patch_size)
        lf = torch.from_numpy(lf_patch).float().unsqueeze(0)
        hf = torch.from_numpy(hf_patch).float().unsqueeze(0)
        return lf, hf


def set_trainable(model, train_head_tail_only: bool):
    """
    Control which parameters are trainable.
    
    Args:
        model: EDSR_X1_Refiner instance
        train_head_tail_only: If True, only head/tail layers are trainable (fine-tuning mode).
                             If False, all parameters are trainable (from-scratch mode).
    """
    for name, p in model.named_parameters():
        if train_head_tail_only:
            p.requires_grad = (name.startswith("head") or name.startswith("tail"))
        else:
            p.requires_grad = True


def compute_metrics(pred, target):
    """
    Compute SSIM and PSNR metrics between prediction and target.
    
    Args:
        pred: Predicted tensor (batch, channels, H, W) in range [0, 1]
        target: Target tensor (batch, channels, H, W) in range [0, 1]
    
    Returns:
        ssim_val: Structural Similarity Index (higher is better, max=1.0)
        psnr_val: Peak Signal-to-Noise Ratio in dB (higher is better)
    """
    # Clamp predictions to valid range [0, 1]
    pred = torch.clamp(pred, 0.0, 1.0)
    target = torch.clamp(target, 0.0, 1.0)
    
    ssim_val = ssim(pred, target, data_range=1.0)
    psnr_val = psnr(pred, target, data_range=1.0)
    
    return ssim_val, psnr_val


if __name__ == "__main__":
    # ==================== STEP 1: Load Pretrained Checkpoint ====================
    print("STEP 1: Loading pretrained 2x EDSR checkpoint from HuggingFace...")
    ckpt_path = hf_hub_download(
        repo_id="eugenesiow/edsr-base",
        filename="pytorch_model_2x.pt",
    )
    state = torch.load(ckpt_path, map_location="cpu")
    print(f"Checkpoint loaded: {len(state)} parameters\n")

    # ==================== STEP 2: Initialize Model ====================
    print("STEP 2: Initializing EDSR_X1_Refiner model...")
    model = EDSR_X1_Refiner(in_ch=1, out_ch=1, n_feats=64, n_resblocks=16)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model initialized with {total_params} parameters")
    
    # Transfer pretrained backbone weights from 2x EDSR to our 1x Refiner
    # (ResBlocks and shared layers), skipping upsampling layers
    print("Loading pretrained backbone weights (64/70 tensors match)...")
    model = load_pretrained_backbone(model, state)
    
    # Fine-tune only head and tail (newly added layers), freeze pretrained ResBlocks
    set_trainable(model, train_head_tail_only=True)
    print(f"Backbone frozen, only head/tail trainable\n")

    # ==================== STEP 3: Setup Device & Optimizer ====================
    print("STEP 3: Setting up device and optimizer...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Model moved to {device}")

    opt = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=3e-4
    )
    num_trainable = len(opt.param_groups[0]['params'])
    print(f"Optimizer: Adam with {num_trainable} trainable parameters\n")

    # ==================== STEP 4: Load Training Data ====================
    print("STEP 4: Loading and preprocessing training data...")
    lf_path = "mri_resolution/train/low_field/sample_001_lowfield.nii"
    hf_path = "mri_resolution/train/high_field/sample_001_highfield.nii"
    
    # Create dataset (loads and aligns volumes, extracts 2D patch coordinates)
    full_dataset = MRIPatchDataset(lf_path, hf_path, patch_size=96, stride=48)
    
    # Split into 80% train, 20% validation
    val_split = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_split
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_split])
    
    # Create DataLoaders (patches extracted on-the-fly during training)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)
    
    print(f"Train patches: {len(train_dataset)}, Val patches: {len(val_dataset)}\n")

    # ==================== STEP 5: Quick Overfit Test ====================
    print("STEP 5: Running overfit test (5 batches to verify gradients flow)...")
    criterion = torch.nn.L1Loss()
    
    model.train()
    overfit_loss = 0.0
    for batch_idx, (lf_batch, hf_batch) in enumerate(train_loader):
        if batch_idx >= 5:
            break
        
        lf_batch = lf_batch.to(device)  # (batch, 1, 96, 96)
        hf_batch = hf_batch.to(device)
        
        opt.zero_grad()
        pred = model(lf_batch)  # Model predicts residual (HF - LF)
        residual = hf_batch - lf_batch  # Target: true residual
        loss = criterion(pred, residual)
        loss.backward()
        opt.step()
        
        overfit_loss += loss.item()
    
    avg_overfit = overfit_loss / 5
    print(f"Overfit test loss: {avg_overfit:.6f} (should decrease towards 0)\n")

    # ==================== STEP 6: Main Training Loop ====================
    print("STEP 6: Starting main training loop...\n")
    num_epochs = 10
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for lf_batch, hf_batch in train_loader:
            lf_batch = lf_batch.to(device)
            hf_batch = hf_batch.to(device)
            
            opt.zero_grad()
            pred_residual = model(lf_batch)  # Model predicts residual (HF - LF)
            residual = hf_batch - lf_batch  # Target: true residual
            loss = criterion(pred_residual, residual)
            loss.backward()
            opt.step()
            
            train_loss += loss.item()
        
        # Validation phase (no gradients)
        model.eval()
        val_loss = 0.0
        val_ssim = 0.0
        val_psnr = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for lf_batch, hf_batch in val_loader:
                lf_batch = lf_batch.to(device)
                hf_batch = hf_batch.to(device)
                
                # Pred residual, reconstruct full HF estimate
                pred_residual = model(lf_batch)
                pred_hf = lf_batch + pred_residual  # Reconstruct: LF + residual = HF
                
                # Calculate loss on residual
                residual = hf_batch - lf_batch
                loss = criterion(pred_residual, residual)
                val_loss += loss.item()
                
                # Calculate SSIM and PSNR on reconstructed HF vs true HF
                batch_ssim, batch_psnr = compute_metrics(pred_hf, hf_batch)
                val_ssim += batch_ssim.item()
                val_psnr += batch_psnr.item()
                
                num_batches += 1
        
        avg_train = train_loss / len(train_loader)
        avg_val_loss = val_loss / num_batches
        avg_val_ssim = val_ssim / num_batches
        avg_val_psnr = val_psnr / num_batches
        print(f"Epoch {epoch+1:2d}/{num_epochs} | Loss: {avg_train:.6f} → {avg_val_loss:.6f} | SSIM: {avg_val_ssim:.4f} | PSNR: {avg_val_psnr:.2f} dB")


