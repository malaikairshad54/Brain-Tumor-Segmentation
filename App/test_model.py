"""
Quick diagnostic script to verify the model is working correctly
"""
import torch
import numpy as np
from unet import UNet2D

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet2D(in_channels=4, out_channels=1).to(device)

MODEL_PATH = r"Model/best_unet2d_brats 3rd(0.78).pth"
state_dict = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state_dict)
model.eval()

print("[OK] Model loaded successfully\n")

# Test 1: Random input (should give random output)
print("Test 1: Random Input")
random_input = torch.randn(1, 4, 192, 192).to(device)
with torch.no_grad():
    logits = model(random_input)
    probs = torch.sigmoid(logits)
print(f"   Logits range: [{logits.min():.4f}, {logits.max():.4f}]")
print(f"   Probs range:  [{probs.min():.4f}, {probs.max():.4f}]")
print()

# Test 2: All zeros (should give low probability)
print("Test 2: All Zeros Input")
zero_input = torch.zeros(1, 4, 192, 192).to(device)
with torch.no_grad():
    logits = model(zero_input)
    probs = torch.sigmoid(logits)
print(f"   Logits range: [{logits.min():.4f}, {logits.max():.4f}]")
print(f"   Probs range:  [{probs.min():.4f}, {probs.max():.4f}]")
print()

# Test 3: Simulated tumor pattern (bright spot in center)
print("Test 3: Simulated Tumor Pattern")
tumor_input = torch.zeros(1, 4, 192, 192)
# Create a bright circular region in the center
center = 96
radius = 20
for i in range(4):
    for y in range(center-radius, center+radius):
        for x in range(center-radius, center+radius):
            if (y-center)**2 + (x-center)**2 < radius**2:
                tumor_input[0, i, y, x] = 5.0  # Bright value

tumor_input = tumor_input.to(device)
with torch.no_grad():
    logits = model(tumor_input)
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    
print(f"   Logits range: [{logits.min():.4f}, {logits.max():.4f}]")
print(f"   Probs range:  [{probs.min():.4f}, {probs.max():.4f}]")
print(f"   Predicted tumor pixels: {preds.sum().item()}")
print()

# Test 4: Check model parameters
print("Test 4: Model Parameter Check")
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"   Total parameters: {total_params:,}")
print(f"   Trainable parameters: {trainable_params:,}")

# Check if any weights are NaN or Inf
has_nan = any(torch.isnan(p).any() for p in model.parameters())
has_inf = any(torch.isinf(p).any() for p in model.parameters())
print(f"   Has NaN weights: {has_nan}")
print(f"   Has Inf weights: {has_inf}")

# Check weight statistics
first_conv_weight = model.enc1.conv[0].weight
print(f"\n   First conv layer weight stats:")
print(f"   Mean: {first_conv_weight.mean():.6f}")
print(f"   Std:  {first_conv_weight.std():.6f}")
print(f"   Min:  {first_conv_weight.min():.6f}")
print(f"   Max:  {first_conv_weight.max():.6f}")

print("\n" + "="*50)
print("DIAGNOSIS:")
print("="*50)

if has_nan or has_inf:
    print("[ERROR] Model has corrupted weights (NaN/Inf)")
elif probs.max() < 0.2:
    print("[ERROR] Model seems broken - even simulated tumor gives low probability")
    print("   Possible causes:")
    print("   1. Wrong model architecture")
    print("   2. Model file is corrupted")
    print("   3. Model was not trained properly")
else:
    print("[OK] Model seems to be working")
    print("   The issue might be:")
    print("   1. The uploaded slice has no tumor")
    print("   2. Preprocessing still doesn't match training exactly")
