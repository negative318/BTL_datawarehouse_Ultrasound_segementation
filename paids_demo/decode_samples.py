import h5py
import os
import numpy as np

image_dir = r"c:\Users\ad\Desktop\New folder (3)\CV\csv_challenge\train\train\images"
label_dir = r"c:\Users\ad\Desktop\New folder (3)\CV\csv_challenge\train\train\labels"

print("Decoding first 10 samples (images and labels)...")

for i in range(10):
    img_filename = f"{i:04d}.h5"
    lbl_filename = f"{i:04d}_label.h5"
    img_path = os.path.join(image_dir, img_filename)
    lbl_path = os.path.join(label_dir, lbl_filename)
    
    img_info = {}
    lbl_info = {}
    
    if os.path.exists(img_path):
        with h5py.File(img_path, 'r') as f:
            for k in f.keys():
                data = f[k][()]
                img_info[k] = {
                    "shape": data.shape if hasattr(data, 'shape') else 'scalar', 
                    "dtype": data.dtype if hasattr(data, 'dtype') else type(data), 
                    "min": np.min(data) if isinstance(data, np.ndarray) and data.size>0 else data, 
                    "max": np.max(data) if isinstance(data, np.ndarray) and data.size>0 else data
                }
                
    if os.path.exists(lbl_path):
        with h5py.File(lbl_path, 'r') as f:
            for k in f.keys():
                data = f[k][()]
                lbl_info[k] = {
                    "shape": data.shape if hasattr(data, 'shape') else 'scalar', 
                    "dtype": data.dtype if hasattr(data, 'dtype') else type(data), 
                    "min": np.min(data) if isinstance(data, np.ndarray) and data.size>0 else data, 
                    "max": np.max(data) if isinstance(data, np.ndarray) and data.size>0 else data
                }
                
    print(f"Sample {i:04d}:")
    print("  Image file:", img_filename)
    for k, v in img_info.items():
        min_v = f"{v['min']:.2f}" if isinstance(v['min'], (float, np.floating)) else v['min']
        max_v = f"{v['max']:.2f}" if isinstance(v['max'], (float, np.floating)) else v['max']
        print(f"    - {k}: shape {v['shape']}, dtype {v['dtype']}, min {min_v}, max {max_v}")
    
    print("  Label file:", lbl_filename)
    for k, v in lbl_info.items():
        min_v = f"{v['min']:.2f}" if isinstance(v['min'], (float, np.floating)) else v['min']
        max_v = f"{v['max']:.2f}" if isinstance(v['max'], (float, np.floating)) else v['max']
        print(f"    - {k}: shape {v['shape']}, dtype {v['dtype']}, min {min_v}, max {max_v}")
    print()
