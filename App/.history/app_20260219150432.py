import os
import io
import base64
import time
import numpy as np
import torch
from flask import Flask, render_template, request
from PIL import Image
import nibabel as nib
from unet import UNet2D

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024  # 1GB max upload (for large medical MRI volumes)

# Create uploads dir if not exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load Model using Relative Path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'Model', 'best_unet2d_brats 3rd(0.78).pth')

model = UNet2D(in_channels=4, out_channels=1).to(device)

print(f"🔄 Loading model from: {MODEL_PATH}")
try:
    if os.path.exists(MODEL_PATH):
        state_dict = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        print("✅ Model loaded successfully!")
    else:
        print(f"❌ Model file not found at {MODEL_PATH}")
except Exception as e:
    print(f"❌ Error loading model: {e}")

def preprocess_image(file_storage, target_size=(192, 192)):
    """
    CRITICAL: Match training preprocessing EXACTLY
    Training used RAW pixel values (0-2399) with NO normalization
    Only center crop to 192x192
    """
    filename = file_storage.filename.lower()
    
    if filename.endswith(('.nii', '.nii.gz')):
        # Handle NIfTI
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], file_storage.filename)
        file_storage.save(temp_path)
        
        try:
            nimg = nib.load(temp_path)
            data = nimg.get_fdata()
            
            # If 3D, take middle slice
            if data.ndim == 3:
                mid_slice = data.shape[2] // 2
                img_slice = data[:, :, mid_slice]
            elif data.ndim == 2:
                img_slice = data
            else:
                img_slice = data[:, :, 0]
            
            # Center crop to 192x192 (EXACTLY like training)
            h, w = img_slice.shape
            if h >= 192 and w >= 192:
                ch = (h - 192) // 2
                cw = (w - 192) // 2
                img_slice = img_slice[ch:ch+192, cw:cw+192]
            else:
                # Resize if smaller
                pil_img = Image.fromarray(img_slice.astype(np.float32))
                pil_img = pil_img.resize(target_size, Image.Resampling.BILINEAR)
                img_slice = np.array(pil_img)
            
            print(f"[NIfTI] {filename} - Min: {img_slice.min():.2f}, Max: {img_slice.max():.2f}, Mean: {img_slice.mean():.2f}")
            
            # Return RAW values (NO normalization)
            return img_slice.astype(np.float32)
            
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
    else:
        # Handle Standard Image (PNG/JPG)
        file_storage.stream.seek(0)
        img = Image.open(file_storage.stream).convert('L')
        
        # Resize to 192x192 or center crop if larger
        if img.size[0] >= 192 and img.size[1] >= 192:
            # Center crop
            left = (img.size[0] - 192) // 2
            top = (img.size[1] - 192) // 2
            img = img.crop((left, top, left + 192, top + 192))
        else:
            img = img.resize(target_size, Image.Resampling.BILINEAR)
        
        img_np = np.array(img).astype(np.float32)
        
        print(f"[Image] {filename} - Min: {img_np.min():.2f}, Max: {img_np.max():.2f}, Mean: {img_np.mean():.2f}")
        
        # Return RAW pixel values (0-255 for images)
        return img_np

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        start_time = time.time()
        
        # Get files
        f_flair = request.files.get('flair')
        f_t1 = request.files.get('t1')
        f_t1ce = request.files.get('t1ce')
        f_t2 = request.files.get('t2')
        
        if not (f_flair and f_t1 and f_t1ce and f_t2):
            return "❌ Missing files! Please upload all 4 modalities."
            
        try:
            # Preprocess all 4 channels
            p_flair = preprocess_image(f_flair)
            p_t1 = preprocess_image(f_t1)
            p_t1ce = preprocess_image(f_t1ce)
            p_t2 = preprocess_image(f_t2)
            
            # Stack: (4, 192, 192)
            img_stack = np.stack([p_flair, p_t1, p_t1ce, p_t2], axis=0)
            
            print(f"\n[Input Stack Stats]")
            print(f"   Shape: {img_stack.shape}")
            print(f"   Min: {img_stack.min():.2f}, Max: {img_stack.max():.2f}")
            print(f"   Mean: {img_stack.mean():.2f}, Std: {img_stack.std():.2f}")
            
            # Add Batch Dim: (1, 4, 192, 192)
            tensor_in = torch.from_numpy(img_stack).float().unsqueeze(0).to(device)
            
            # Inference
            with torch.no_grad():
                logits = model(tensor_in)
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
            
            print(f"\n🤖 Model Output:")
            print(f"   Logits - Min: {logits.min().item():.4f}, Max: {logits.max().item():.4f}, Mean: {logits.mean().item():.4f}")
            print(f"   Probs  - Min: {probs.min().item():.4f}, Max: {probs.max().item():.4f}, Mean: {probs.mean().item():.4f}")
            print(f"   Preds  - Positive pixels: {preds.sum().item()}")
            
            # Process Result
            pred_mask = preds.squeeze().cpu().numpy() # (192, 192)
            
            # Calculate tumor %
            tumor_pixels = np.sum(pred_mask)
            total_pixels = pred_mask.size
            tumor_pct = (tumor_pixels / total_pixels) * 100
            
            # Create Visualization (Overlay)
            # Use FLAIR for display
            flair_disp = p_flair
            f_min, f_max = flair_disp.min(), flair_disp.max()
            if f_max > f_min:
                flair_disp = (flair_disp - f_min) / (f_max - f_min)
            flair_uint8 = (flair_disp * 255).astype(np.uint8)
            
            # Create RGB Base
            base_img = Image.fromarray(flair_uint8).convert("RGBA")
            
            # Create Mask Layer (Red)
            # Mask needs to be resized to match base if base wasn't 192? No, we resized everything to 192.
            mask_uint8 = (pred_mask * 255).astype(np.uint8)
            mask_img = Image.fromarray(mask_uint8).convert("L")
            
            # Create a solid color layer for the mask
            red_layer = Image.new("RGBA", base_img.size, (255, 50, 50, 100)) # Semi-transparent Red
            
            # Composite
            mask_overlay = Image.composite(red_layer, Image.new("RGBA", base_img.size, (0,0,0,0)), mask_img)
            final_comp = Image.alpha_composite(base_img, mask_overlay)
            
            # Save to buffer
            buf = io.BytesIO()
            final_comp.save(buf, format='PNG')
            img_b64 = base64.b64encode(buf.getvalue()).decode('ascii')
            
            inference_time = round(time.time() - start_time, 3)
            
            return render_template('analyze.html', 
                                   prediction_image=img_b64,
                                   tumor_percentage=round(tumor_pct, 2),
                                   inference_time=inference_time,
                                   has_tumor=(tumor_pixels > 0))
                                   
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"❌ Error during processing: {str(e)}"

    return render_template('analyze.html', prediction_image=None)

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    print("🚀 Starting Flask Server...")
    print(f"📂 Model Path: {MODEL_PATH}")
    app.run(debug=True, port=5000)
