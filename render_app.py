import gradio as gr
import requests
import numpy as np
import io
import time
import pandas as pd
from PIL import Image
import plotly.graph_objects as go

# LOCAL NGROK URL
LOCAL_API_URL = "https://cicada-logical-virtually.ngrok-free.app"

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def mask_to_rgb(mask_np):
    mask_np = np.array(mask_np)
    h, w = mask_np.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[mask_np == 1] = [255, 50, 50] # Red
    rgb[mask_np == 2] = [50, 255, 50] # Green
    return Image.fromarray(rgb)

# GLOBAL HISTORY
inference_history = {
    "times": [],
    "scores": [],
    "timestamps": [],
    "results": [],
    "masks_L": [],
    "masks_T": []
}

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
        diag_result = "POSITIVE" if prob_score >= 0.5 else "NEGATIVE"
        
        m_L = mask_to_rgb(data["mask_L"])
        m_T = mask_to_rgb(data["mask_T"])

        inference_history["times"].append(duration)
        inference_history["scores"].append(prob_score)
        inference_history["timestamps"].append(time.strftime("%Y-%m-%d %H:%M:%S"))
        inference_history["results"].append(diag_result)
        inference_history["masks_L"].append(m_L)
        inference_history["masks_T"].append(m_T)

        return m_L, m_T, f"Diagnosis: {diag_result} (Prob: {prob_score:.4f})"
    except Exception as e:
        return None, None, f"Connection error: {str(e)}"

def update_history():
    if not inference_history["timestamps"]:
        return pd.DataFrame(columns=["ID", "Timestamp", "Result", "Prob Score", "Visualize"])
    df = pd.DataFrame({
        "ID": range(len(inference_history["timestamps"])),
        "Timestamp": inference_history["timestamps"],
        "Result": inference_history["results"],
        "Prob Score": [f"{s:.4f}" for s in inference_history["scores"]],
        "Visualize": ["👁️ View Details"] * len(inference_history["timestamps"])
    })
    return df.iloc[::-1]

def handle_history_click(evt: gr.SelectData):
    if evt.index[1] != 4:
        return gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
    
    total = len(inference_history["timestamps"])
    row_id = total - 1 - evt.index[0]
    
    if row_id < 0 or row_id >= total:
        return gr.update(visible=False), None, None, "", None
    
    m_L = inference_history["masks_L"][row_id]
    m_T = inference_history["masks_T"][row_id]
    res = inference_history["results"][row_id]
    score = inference_history["scores"][row_id]
    lat = inference_history["times"][row_id]
    
    # Create Gauge specifically for this record
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score,
        title = {'text': f"Confidence: {res}"},
        gauge = {
            'axis': {'range': [0, 1]},
            'bar': {'color': "red" if res=="POSITIVE" else "green"},
            'steps': [
                {'range': [0, 0.5], 'color': "#f0f0f0"},
                {'range': [0.5, 1], 'color': "#d0d0d0"}
            ],
            'threshold': {'line': {'color': "black", 'width': 4}, 'value': 0.5}
        }
    ))
    fig_gauge.update_layout(height=250, margin=dict(l=30, r=30, t=50, b=20))

    info = f"ID: #{row_id} | Time: {inference_history['timestamps'][row_id]} | Latency: {lat:.3f}s"
    return gr.update(visible=True), m_L, m_T, info, fig_gauge

with gr.Blocks(title="Ultrasound AI System", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🏥 Ultrasound AI System")
    
    with gr.Tabs() as tabs:
        # --- TAB 1: INFERENCE ---
        with gr.TabItem("🔍 Inference", id=0):
            with gr.Row():
                with gr.Column():
                    in_long = gr.Image(label="1. Input Longitudinal", type="numpy")
                    in_trans = gr.Image(label="2. Input Transverse", type="numpy")
                    btn = gr.Button("🧠 Start Processing", variant="primary")
                with gr.Column():
                    out_long = gr.Image(label="Predicted Longitudinal Mask")
                    out_trans = gr.Image(label="Predicted Transverse Mask")
                    out_cls = gr.Textbox(label="Status")
            btn.click(fn=predict_remote, inputs=[in_long, in_trans], outputs=[out_long, out_trans, out_cls])

        # --- TAB 2: HISTORY (Integrating Metrics Dashboard here) ---
        with gr.TabItem("📜 History", id=1) as tab_hist:
            gr.Markdown("### 📋 Diagnosis History & Performance Analytics")
            
            history_df = gr.Dataframe(
                headers=["ID", "Timestamp", "Result", "Prob Score", "Visualize"],
                datatype=["number", "str", "str", "str", "str"],
                label="Diagnosis History Table",
                interactive=False
            )
            
            with gr.Column(visible=False) as detail_area:
                gr.Markdown("---")
                gr.Markdown("### 📊 Metrics Dashboard for this Inference")
                close_btn = gr.Button("❌ Close Detail View", size="sm")
                
                with gr.Row():
                    with gr.Column(scale=2):
                        with gr.Row():
                            hist_long = gr.Image(label="Longitudinal Mask Preview")
                            hist_trans = gr.Image(label="Transverse Mask Preview")
                        hist_status = gr.Textbox(label="Inference Performance Info")
                    with gr.Column(scale=1):
                        # Gauge plot specifically for this record
                        hist_plot = gr.Plot(label="Confidence Gauge")

    # --- EVENTS ---
    tab_hist.select(fn=update_history, outputs=history_df)
    history_df.select(
        fn=handle_history_click, 
        outputs=[detail_area, hist_long, hist_trans, hist_status, hist_plot]
    )
    close_btn.click(fn=lambda: gr.update(visible=False), outputs=detail_area)

if __name__ == "__main__":
    # --- WARM-UP ---
    print("♨️ Initializing AI System...")
    _ = go.Figure(go.Indicator(value=0.5))
    print("✅ System Ready.")

    import os
    port = int(os.environ.get("PORT", 7860))
    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=port, show_error=True)
