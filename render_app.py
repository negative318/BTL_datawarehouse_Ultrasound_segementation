import gradio as gr
import requests
import numpy as np
import io
import time
from PIL import Image
import glob
import os

# API URL (Configurable via Environment Variable for Deployment)
LOCAL_API_URL = os.environ.get("API_URL", "https://cicada-logical-virtually.ngrok-free.app")

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def mask_to_rgb(mask_np):
    mask_np = np.array(mask_np)
    h, w = mask_np.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[mask_np == 1] = [255, 50, 50] # Red
    rgb[mask_np == 2] = [50, 255, 50] # Green
    return Image.fromarray(rgb)

def predict_remote(long_img, trans_img):
    start_time = time.time()
    def img_to_bytes(img):
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, format="JPEG")
        return buf.getvalue()

    files = {
        "long_file": ("long.jpg", img_to_bytes(long_img), "image/jpeg"),
        "trans_file": ("trans.jpg", img_to_bytes(trans_img), "image/jpeg")
    }

    try:
        response = requests.post(f"{LOCAL_API_URL}/predict", files=files, timeout=60)
        data = response.json()
        duration = time.time() - start_time
        raw_score = data["cls_score"]
        prob_score = float(sigmoid(raw_score) if (raw_score < 0 or raw_score > 1) else raw_score)
        
        if prob_score >= 0.5:
            diag_result = "POSITIVE"
            certainty = prob_score * 100
        else:
            diag_result = "NEGATIVE"
            certainty = (1 - prob_score) * 100
        
        mask_L_np = np.array(data["mask_L"])
        mask_T_np = np.array(data["mask_T"])
        
        m_L = mask_to_rgb(mask_L_np)
        m_T = mask_to_rgb(mask_T_np)

        status_text = (
            f"🩺 Diagnosis: {diag_result}\n"
            f"🎯 Confidence: {certainty:.2f}%\n"
            f"⏱️ Time: {duration:.2f}s"
        )

        return m_L, m_T, status_text
    except Exception as e:
        return None, None, f"Connection error: {str(e)}"

with gr.Blocks(title="Ultrasound AI System") as demo:
    gr.Markdown("# 🏥 Ultrasound AI System")
    
    with gr.Tabs() as tabs:
        with gr.TabItem("🏠 Project Overview"):
            gr.Markdown("## 🚀 Carotid Plaque Segmentation & Vulnerability Assessment")
            gr.Markdown(
                "Welcome to the **Ultrasound AI System**. This application provides an end-to-end deep learning "
                "pipeline designed to assist medical professionals in analyzing carotid ultrasound images."
            )
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### ✨ Core Capabilities")
                    gr.Markdown(
                        "1. **Dual-View Analysis:** Simultaneously processes both **Longitudinal** (long-axis) and **Transverse** (short-axis) ultrasound scans to gain a complete understanding of the plaque.\n"
                        "2. **Precise Segmentation:** Automatically identifies and masks the exact boundaries of carotid plaques, replacing tedious manual drawing.\n"
                        "3. **Risk Classification:** Predicts whether the identified plaque is **High Risk (Vulnerable)** or **Low Risk (Stable)** based on the 2024 Plaque-RADS guidelines."
                    )
                with gr.Column():
                    gr.Markdown("### 🛠️ How to Use This System")
                    gr.Markdown(
                        "- Navigate to the **🔍 Inference** tab.\n"
                        "- Upload the patient's Longitudinal and Transverse ultrasound images into the respective boxes.\n"
                        "- Click the **Start Processing** button.\n"
                        "- The AI will return the predicted segmentation masks along with the diagnostic confidence score and estimated inference latency in real-time."
                    )
                    
            gr.Markdown("---")
            gr.Markdown("## 🗂️ Training Dataset: Ultrasound Plaque Dataset")
            gr.Markdown(
                "This model is trained on a comprehensive **Carotid Plaque Segmentation and Vulnerability Assessment in Ultrasound** dataset. "
                "The data is collected from **Multiple Ultrasound Systems and Multiple Hospital Centers**, introducing substantial variability in imaging appearance and clinical practice."
            )
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 📊 Dataset Statistics")
                    gr.Markdown(
                        "- **Total Cases:** 1000 cases\n"
                        "- **Data Modality:** Ultrasound (Longitudinal & Transverse views)\n"
                        "- **Labeled Data:** 200 cases (20%)\n"
                        "- **Unlabeled Data:** 800 cases (80%)\n"
                        "- **Class Distribution:** Low Risk : High Risk ≈ 4 : 1 (Overall) / 1 : 1 (Labeled Subset)"
                    )
                with gr.Column():
                    gr.Markdown("### 🏷️ Annotations & Guidelines")
                    gr.Markdown(
                        "- **Segmentation Masks:** Provided for both Longitudinal and Transverse views.\n"
                        "- **Risk Stratification:** Follows the **2024 Plaque-RADS** guideline.\n"
                        "  - **Low Risk (NEGATIVE):** RADS = 2\n"
                        "  - **High Risk (POSITIVE):** RADS = 3–4\n"
                        "- **Preprocessing:** All images are center-aligned, zero-padded, and resized to 512 pixels on the longer side."
                    )
            
            gr.Markdown("---")
            gr.Markdown("### 📸 Sample Ultrasound Images")
            gr.Markdown("Below are representative samples of the dataset:")
            
            sample_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gradio_samples")
            long_images = sorted(glob.glob(os.path.join(sample_dir, "*_long.png")))
            trans_images = sorted(glob.glob(os.path.join(sample_dir, "*_trans.png")))
            
            for i in range(0, len(long_images), 2):
                with gr.Row():
                    # Pair 1
                    with gr.Group():
                        sample_id_1 = os.path.basename(long_images[i]).split('_')[1]
                        gr.Markdown(f"### 🩺 Case {sample_id_1}")
                        with gr.Row():
                            gr.Image(value=long_images[i], label="Longitudinal", interactive=False)
                            gr.Image(value=trans_images[i], label="Transverse", interactive=False)
                            
                    # Pair 2
                    if i + 1 < len(long_images):
                        with gr.Group():
                            sample_id_2 = os.path.basename(long_images[i+1]).split('_')[1]
                            gr.Markdown(f"### 🩺 Case {sample_id_2}")
                            with gr.Row():
                                gr.Image(value=long_images[i+1], label="Longitudinal", interactive=False)
                                gr.Image(value=trans_images[i+1], label="Transverse", interactive=False)

            gr.Markdown("---")
            gr.Markdown("<div style='text-align: center; color: gray;'><i>Developed as a Final Project for the Data Warehouse course. For research and educational purposes only.</i></div>")

        with gr.TabItem("🔍 Inference"):
            with gr.Row():
                with gr.Column():
                    in_long = gr.Image(label="1. Input Longitudinal", type="numpy")
                    in_trans = gr.Image(label="2. Input Transverse", type="numpy")
                    btn = gr.Button("🧠 Start Processing", variant="primary")
                with gr.Column():
                    out_long = gr.Image(label="Predicted Longitudinal Mask")
                    out_trans = gr.Image(label="Predicted Transverse Mask")
                    out_cls = gr.Textbox(label="Inference Metrics", lines=4)

            btn.click(fn=predict_remote, inputs=[in_long, in_trans], outputs=[out_long, out_trans, out_cls])

        with gr.TabItem("📈 Evaluation Metrics"):
            gr.Markdown("## 📈 System Evaluation Metrics")
            gr.Markdown("Below are the benchmark results of the AI model on the test dataset (Simulated for Demo).")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 🩻 Segmentation Performance")
                    gr.Dataframe(
                        headers=["View", "Mean Dice", "Mean IoU", "HD95 (mm)"],
                        value=[
                            ["Longitudinal (Mask L)", "0.895", "0.821", "2.14"],
                            ["Transverse (Mask T)", "0.872", "0.789", "2.45"]
                        ],
                        interactive=False
                    )
                with gr.Column():
                    gr.Markdown("### ⚕️ Classification Performance")
                    gr.Dataframe(
                        headers=["Metric", "Score"],
                        value=[
                            ["Accuracy", "92.5%"],
                            ["AUC-ROC", "0.954"],
                            ["Sensitivity (Recall)", "89.2%"],
                            ["Specificity", "94.1%"],
                            ["F1-Score", "0.910"]
                        ],
                        interactive=False
                    )

if __name__ == "__main__":
    # --- WARM-UP ---
    print("✅ System Ready.")

    import os
    port = int(os.environ.get("PORT", 7860))

    print(f"🚀 Starting Render Frontend on port {port}...")
    demo.queue() 
    
    demo.launch(
        server_name="0.0.0.0", 
        server_port=port,
        show_error=True,
        theme=gr.themes.Soft()
    )
