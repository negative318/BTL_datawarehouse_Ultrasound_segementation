import h5py
import os
import matplotlib.image as mpimg

image_dir = r"c:\Users\pc\OneDrive\Desktop\BTL_DataWareHouse\train\train\images"
out_dir = r"c:\Users\pc\OneDrive\Desktop\BTL_DataWareHouse\paids_demo\gradio_samples"

os.makedirs(out_dir, exist_ok=True)

for i in range(10):
    img_path = os.path.join(image_dir, f"{i:04d}.h5")
    if os.path.exists(img_path):
        with h5py.File(img_path, 'r') as f:
            mpimg.imsave(os.path.join(out_dir, f"sample_{i:04d}_long.png"), f['long_img'][()], cmap='gray')
            mpimg.imsave(os.path.join(out_dir, f"sample_{i:04d}_trans.png"), f['trans_img'][()], cmap='gray')
            print(f"Extracted {i:04d}")
