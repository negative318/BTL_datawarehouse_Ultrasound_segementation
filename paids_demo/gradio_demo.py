import os
import torch
import torch.nn.functional as F
import numpy as np

try:
    import gradio as gr
except ImportError:
    print("Please install gradio: uv pip install gradio")
    exit(1)

from unet import UNetTwoView

# 1. Global Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Server initializing - Model will run on: {device}")

model_cache = {"model": None}

def load_model():
    """Lazily loads the model upon first request so startup is instant."""
    if model_cache["model"] is not None:
        return model_cache["model"]
    
    print("Loading weights into VRAM...")
    model = UNetTwoView(in_chns=1, seg_class_num=3, cls_class_num=1, img_size=256)
    weight_path = r"best.pth"
    
    if os.path.exists(weight_path):
        state_dict = torch.load(weight_path, map_location=device, weights_only=False)
        if "model" in state_dict:
            state_dict = state_dict["model"]
        elif "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
            
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:] if k.startswith("module.") else k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict, strict=False)
        print("Weights successfully loaded.")
    else:
        print("WARNING: best.pth not found! Using random weights.")
        
    model.to(device)
    model.eval()
    model_cache["model"] = model
    return model

# 2. Image Processing
def preprocess_image(img_np):
    """Converts a raw uploaded numpy image into a batched Tensor."""
    if img_np is None:
        return torch.zeros(1, 1, 256, 256, device=device)
        
    # If the image was uploaded as RGB, convert to grayscale format manually just in case
    if len(img_np.shape) == 3 and img_np.shape[2] >= 3:
        img_np = np.mean(img_np[:, :, :3], axis=2)
    
    t = torch.from_numpy(img_np).float().unsqueeze(0).unsqueeze(0).to(device)
    t = F.interpolate(t, size=(256, 256), mode='bilinear', align_corners=False)
    
    if t.max() > 1.0:
        t = t / 255.0
    return t

def mask_to_rgb(mask_np):
    """Maps segmentation class indices to visible RGB colors."""
    h, w = mask_np.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Class 1: Red
    rgb[mask_np == 1] = [255, 50, 50]
    # Class 2: Green
    rgb[mask_np == 2] = [50, 255, 50]
    # Class 0: Black Background
    
    return rgb

# 3. Main Gradio Pipeline
def predict_ultrasound(long_img, trans_img):
    model = load_model()
    
    pt_long = preprocess_image(long_img)
    pt_trans = preprocess_image(trans_img)
    
    with torch.no_grad():
        seg_logits_L, seg_logits_T, cls_logits = model(pt_long, pt_trans, need_fp=False)
    
    mask_L = torch.argmax(seg_logits_L, dim=1).squeeze().cpu().numpy()
    mask_T = torch.argmax(seg_logits_T, dim=1).squeeze().cpu().numpy()
    
    # Overlay logic / Color mapping
    color_L = mask_to_rgb(mask_L)
    color_T = mask_to_rgb(mask_T)
    
    cls_score = cls_logits.item()
    status = f"Raw Logit: {cls_score:.4f}\nDiagnosis: {'POSITIVE' if cls_score >= 0.5 else 'NEGATIVE'}"
    
    return color_L, color_T, status

# 4. Gradio UI Interface
with gr.Blocks(theme=gr.themes.Soft(), title="Ultrasound AI") as demo:
    gr.Markdown("# 🩺 Ultrasound Dual-View Diagnostic AI")
    gr.Markdown("Upload a patient's **Longitudinal** and **Transverse** ultrasound scans to generate AI segmentation masks and predict gallbladder status locally on your GPU.")
    
    with gr.Row():
        with gr.Column():
            in_long = gr.Image(label="1. Input Longitudinal", type="numpy", image_mode="L")
            in_trans = gr.Image(label="2. Input Transverse", type="numpy", image_mode="L")
            btn = gr.Button("🧠 Segment & Predict", variant="primary", size="lg")
            
        with gr.Column():
            out_long = gr.Image(label="Predicted Longitudinal Mask")
            out_trans = gr.Image(label="Predicted Transverse Mask")
            out_cls = gr.Textbox(label="Classification Status")
            
    btn.click(fn=predict_ultrasound, inputs=[in_long, in_trans], outputs=[out_long, out_trans, out_cls])

if __name__ == "__main__":
    print("Launching Gradio Data Server...")
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
