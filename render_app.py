import gradio as gr
import requests
import numpy as np
import io
from PIL import Image

# LOCAL NGROK URL (Update this whenever you restart Ngrok on your local machine)
LOCAL_API_URL = "https://cicada-logical-virtually.ngrok-free.app"

def mask_to_rgb(mask_np):
    mask_np = np.array(mask_np)
    h, w = mask_np.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[mask_np == 1] = [255, 50, 50] # Red
    rgb[mask_np == 2] = [50, 255, 50] # Green
    return rgb

def predict_remote(long_img, trans_img):
    # Convert image to bytes for internet transmission
    def img_to_bytes(img):
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, format="JPEG")
        return buf.getvalue()

    files = {
        "long_file": ("long.jpg", img_to_bytes(long_img), "image/jpeg"),
        "trans_file": ("trans.jpg", img_to_bytes(trans_img), "image/jpeg")
    }

    try:
        response = requests.post(f"{LOCAL_API_URL}/predict", files=files)
        data = response.json()
        
        color_L = mask_to_rgb(data["mask_L"])
        color_T = mask_to_rgb(data["mask_T"])
        cls_score = data["cls_score"]
        status = f"Diagnosis: {'POSITIVE' if cls_score >= 0.5 else 'NEGATIVE'} ({cls_score:.4f})"
        
        return color_L, color_T, status
    except Exception as e:
        return None, None, f"Connection error to Local API: {str(e)}"

with gr.Blocks(title="Ultrasound AI - Cloud") as demo:
    gr.Markdown("# ☁️ Ultrasound AI (Render Cloud Frontend)")
    gr.Markdown("This interface runs on Render, while computation is processed on your Local Machine.")
    
    with gr.Row():
        with gr.Column():
            in_long = gr.Image(label="1. Input Longitudinal", type="numpy")
            in_trans = gr.Image(label="2. Input Transverse", type="numpy")
            btn = gr.Button("🧠 Send to Local for Processing", variant="primary")
            
        with gr.Column():
            out_long = gr.Image(label="Predicted Longitudinal Mask")
            out_trans = gr.Image(label="Predicted Transverse Mask")
            out_cls = gr.Textbox(label="Status")
            
    btn.click(fn=predict_remote, inputs=[in_long, in_trans], outputs=[out_long, out_trans, out_cls])

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 7860))
    
    print(f"🚀 Starting Render Frontend on port {port}...")
    demo.queue() 
    
    demo.launch(
        server_name="0.0.0.0", 
        server_port=port,
        show_error=True
    )
