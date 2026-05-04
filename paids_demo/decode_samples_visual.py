import h5py
import os
import numpy as np
import matplotlib.pyplot as plt

image_dir = r"c:\Users\ad\Desktop\New folder (3)\CV\csv_challenge\train\train\images"
label_dir = r"c:\Users\ad\Desktop\New folder (3)\CV\csv_challenge\train\train\labels"
out_dir = r"c:\Users\ad\Desktop\New folder (3)\CV\csv_challenge\decoded_samples"

os.makedirs(out_dir, exist_ok=True)

print("Decoding first 2 samples into images...")

for i in range(10):
    img_filename = f"{i:04d}.h5"
    lbl_filename = f"{i:04d}_label.h5"
    img_path = os.path.join(image_dir, img_filename)
    lbl_path = os.path.join(label_dir, lbl_filename)
    
    long_img = trans_img = long_mask = trans_mask = None
    cls_label = None

    if os.path.exists(img_path):
        with h5py.File(img_path, 'r') as f:
            long_img = f['long_img'][()]
            trans_img = f['trans_img'][()]
            
    if os.path.exists(lbl_path):
        with h5py.File(lbl_path, 'r') as f:
            long_mask = f['long_mask'][()]
            trans_mask = f['trans_mask'][()]
            cls_label = f['cls'][()]

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle(f"Sample {i:04d} - Class: {cls_label}")
    
    axs[0, 0].imshow(long_img, cmap='gray')
    axs[0, 0].set_title("Longitudinal Image")
    axs[0, 0].axis('off')
    
    axs[0, 1].imshow(long_mask, cmap='jet', alpha=0.5)
    axs[0, 1].set_title("Longitudinal Mask")
    axs[0, 1].axis('off')
    
    axs[1, 0].imshow(trans_img, cmap='gray')
    axs[1, 0].set_title("Transverse Image")
    axs[1, 0].axis('off')
    
    axs[1, 1].imshow(trans_mask, cmap='jet', alpha=0.5)
    axs[1, 1].set_title("Transverse Mask")
    axs[1, 1].axis('off')
    
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"sample_{i:04d}.png")
    plt.savefig(out_path)
    plt.close()
    
    print(f"Saved {out_path}")
