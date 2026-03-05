"""
Verify prediction accuracy by comparing with ground truth
"""
import numpy as np
import nibabel as nib
import torch
from unet import UNet2D
from pathlib import Path

# Configuration
PATIENT_ID = "brats2021_00077"
PATIENT_DIR = Path(r"D:\path\to\your\brats\dataset") / PATIENT_ID  # UPDATE THIS PATH
MODEL_PATH = r"Model/best_unet2d_brats 3rd(0.78).pth"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
model = UNet2D(in_channels=4, out_channels=1).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

print(f"Loading patient: {PATIENT_ID}")

# Load all modalities
flair = nib.load(PATIENT_DIR / f"{PATIENT_ID}_flair.nii.gz").get_fdata()
t1 = nib.load(PATIENT_DIR / f"{PATIENT_ID}_t1.nii.gz").get_fdata()
t1ce = nib.load(PATIENT_DIR / f"{PATIENT_ID}_t1ce.nii.gz").get_fdata()
t2 = nib.load(PATIENT_DIR / f"{PATIENT_ID}_t2.nii.gz").get_fdata()
seg = nib.load(PATIENT_DIR / f"{PATIENT_ID}_seg.nii.gz").get_fdata()

# Get middle slice
z = flair.shape[2] // 2
print(f"Using slice: {z}")

# Stack modalities (MATCH TRAINING)
img_slice = np.stack([
    flair[:, :, z],
    t1[:, :, z],
    t1ce[:, :, z],
    t2[:, :, z]
], axis=0)

# Center crop to 192x192
def center_crop(x, size=192):
    h, w = x.shape[-2:]
    ch = (h - size) // 2
    cw = (w - size) // 2
    return x[..., ch:ch+size, cw:cw+size]

img_slice = center_crop(img_slice)
seg_slice = center_crop(seg[:, :, z])

# Ground truth (binary)
gt_mask = (seg_slice > 0).astype(np.float32)

# Inference
tensor_in = torch.from_numpy(img_slice).float().unsqueeze(0).to(device)
with torch.no_grad():
    logits = model(tensor_in)
    probs = torch.sigmoid(logits)
    pred_mask = (probs > 0.5).float().squeeze().cpu().numpy()

# Calculate Dice Score
intersection = (pred_mask * gt_mask).sum()
union = pred_mask.sum() + gt_mask.sum()
dice = (2 * intersection) / (union + 1e-6)

# Statistics
gt_tumor_pixels = gt_mask.sum()
pred_tumor_pixels = pred_mask.sum()
gt_tumor_pct = (gt_tumor_pixels / gt_mask.size) * 100
pred_tumor_pct = (pred_tumor_pixels / pred_mask.size) * 100

print("\n" + "="*50)
print("VERIFICATION RESULTS")
print("="*50)
print(f"Ground Truth Tumor: {gt_tumor_pixels:.0f} pixels ({gt_tumor_pct:.2f}%)")
print(f"Predicted Tumor:    {pred_tumor_pixels:.0f} pixels ({pred_tumor_pct:.2f}%)")
print(f"\nDice Score: {dice:.4f}")

if dice > 0.7:
    print("\n✅ EXCELLENT prediction (Dice > 0.7)")
elif dice > 0.5:
    print("\n✅ GOOD prediction (Dice > 0.5)")
elif dice > 0.3:
    print("\n⚠️ MODERATE prediction (Dice > 0.3)")
else:
    print("\n❌ POOR prediction (Dice < 0.3)")

print("\nNote: Update PATIENT_DIR path in this script to run verification")
