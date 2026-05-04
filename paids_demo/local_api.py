import os
import torch
import torch.nn.functional as F
import numpy as np
from fastapi import FastAPI, UploadFile, File
from unet import UNetTwoView
import io
from PIL import Image
import uvicorn

app = FastAPI()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNetTwoView(in_chns=1, seg_class_num=3, cls_class_num=1, img_size=256)
weight_path = "best.pth"

if os.path.exists(weight_path):
    state_dict = torch.load(weight_path, map_location=device, weights_only=False)
    if "model" in state_dict: state_dict = state_dict["model"]
    new_state_dict = { (k[7:] if k.startswith("module.") else k): v for k, v in state_dict.items() }
    model.load_state_dict(new_state_dict, strict=False)
    print("✅ Model loaded successfully on Local.")
model.to(device).eval()

def preprocess(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert('L')
    img_np = np.array(img)
    t = torch.from_numpy(img_np).float().unsqueeze(0).unsqueeze(0).to(device)
    t = F.interpolate(t, size=(256, 256), mode='bilinear', align_corners=False)
    return t / 255.0 if t.max() > 1.0 else t

@app.post("/predict")
async def predict(long_file: UploadFile = File(...), trans_file: UploadFile = File(...)):
    # Read files sent from the remote client (e.g., Render)
    pt_long = preprocess(await long_file.read())
    pt_trans = preprocess(await trans_file.read())
    
    with torch.no_grad():
        seg_L, seg_T, cls_logits = model(pt_long, pt_trans)
    
    mask_L = torch.argmax(seg_L, dim=1).squeeze().cpu().numpy().tolist()
    mask_T = torch.argmax(seg_T, dim=1).squeeze().cpu().numpy().tolist()
    
    return {
        "mask_L": mask_L,
        "mask_T": mask_T,
        "cls_score": cls_logits.item()
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
