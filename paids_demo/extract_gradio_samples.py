import h5py
import os
import matplotlib.image as mpimg

# Paths
image_dir = r"c:\Users\pc\OneDrive\Desktop\BTL_DataWareHouse\train\train\images"
out_dir = r"c:\Users\pc\OneDrive\Desktop\BTL_DataWareHouse\gradio_samples"

os.makedirs(out_dir, exist_ok=True)

# Select Sample 0
i = 0
img_filename = f"{i:04d}.h5"
img_path = os.path.join(image_dir, img_filename)

if os.path.exists(img_path):
    print(f"Decoding {img_path} for Gradio demo...")
    with h5py.File(img_path, 'r') as f:
        long_img = f['long_img'][()]
        trans_img = f['trans_img'][()]
        
        # Save as PNG
        long_out = os.path.join(out_dir, f"sample_{i:04d}_long.png")
        trans_out = os.path.join(out_dir, f"sample_{i:04d}_trans.png")
        
        # Normalize to 0-1 if needed, but imsave handles it or use cmap='gray'
        # png saving with imsave
        mpimg.imsave(long_out, long_img, cmap='gray')
        mpimg.imsave(trans_out, trans_img, cmap='gray')
        
        print(f"✅ Saved Longitudinal View: {long_out}")
        print(f"✅ Saved Transverse View: {trans_out}")
else:
    print(f"Error: {img_path} not found.")
