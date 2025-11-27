import streamlit as st
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import os
import csv
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import pandas as pd

# ================== CONFIG ==================
st.set_page_config(page_title="Hospital AI Panel", layout="wide")

# ================== HOSPITAL THEME ==================
st.markdown("""
<style>
/* ======== GLOBAL THEME ======== */
body {
    background: linear-gradient(135deg, #eaf3ff, #ffffff);
    font-family: 'Segoe UI', sans-serif;
}

.main {
    background-color: #ffffff;
    padding: 15px;
}

h1, h2, h3 {
    color: #0b3d91;
}

/* ======== SIDEBAR ======== */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b3d91, #1e5ed8);
}

[data-testid="stSidebar"] * {
    color: white !important;
}

/* ======== BUTTONS ======== */
.stButton>button {
    background: linear-gradient(90deg, #0b3d91, #1e73be);
    color: white;
    border-radius: 25px;
    padding: 10px 25px;
    border: none;
    font-weight: 600;
    transition: 0.3s ease;
}

.stButton>button:hover {
    transform: scale(1.03);
    background: linear-gradient(90deg, #1e73be, #0b3d91);
}

/* ======== CARDS ======== */
.card {
    background: #f5f9ff;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 10px 25px rgba(0,0,0,0.08);
    margin-bottom: 20px;
    color: #1a1a1a !important; /* ✅ FIX: Doctor profile text now dark & visible */
}

.sidebar-profile {
    color: #000000 !important;  /* ✅ Force dark text */
    background: #ffffff !important;
}

.sidebar-profile b {
    color: #000000 !important;
}

.sidebar-profile * {
    color: #000000 !important; /* Ensures all inner text visible */
}

.metric-box {
    background: white;
    border-radius: 12px;
    padding: 15px;
    text-align: center;
    box-shadow: 0 5px 15px rgba(0,0,0,0.05);
}
}

.metric-box {
    background: white;
    border-radius: 12px;
    padding: 15px;
    text-align: center;
    box-shadow: 0 5px 15px rgba(0,0,0,0.05);
}

/* Progress bar aesthetic */
progress {
    height: 20px;
    border-radius: 10px;
}

/* Upload area */
[data-testid="stFileUploader"] {
    border: 2px dashed #0b3d91;
    padding: 20px;
    border-radius: 15px;
}

/* ===== DARK MODE ===== */
.dark-mode body {
    background: #121212;
    color: white;
}

.dark-mode .main {
    background: #1e1e1e;
}

/* Responsive layout */
@media (max-width: 768px) {
  .main {
    padding: 5px;
  }
}
</style>
""", unsafe_allow_html=True)

# ================== AUTH SYSTEM ==================
USERS = {
    "doctor1": "pass123",
    "admin": "admin@123"
}

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.title("🔐 Doctor Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username in USERS and USERS[username] == password:
            st.session_state.logged_in = True
            st.session_state.user = username
            st.success("Login successful ✅")
            st.rerun()
        else:
            st.error("Invalid credentials ❌")
    st.stop()

# ================== SIDEBAR NAV ==================
st.sidebar.markdown(f"""
<div style='display:flex;align-items:center;gap:10px;margin-bottom:15px;'>
<img src='https://img.icons8.com/color/96/hospital-3.png' width='35'/>
<h3 style='margin:0;color:white;'>Hospital AI Panel</h3>
</div>
<div class='card sidebar-profile'>
<b>Name:</b> Dr. {st.session_state.user}<br>
<b>Department:</b> Oncology<br>
<b>Status:</b> Online ✅
</div>
""", unsafe_allow_html=True)
page = st.sidebar.radio("Navigate", [
    "Diagnosis Panel",
    "Patient Database",
    "Analytics Dashboard"
])

# ================= DEVICE =================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
best_params = {'dropout': 0.4995627325190859}

# ================= MODEL LOAD =================
@st.cache_resource
def load_model():
    weights = EfficientNet_B2_Weights.DEFAULT
    model = efficientnet_b2(weights=weights)
    model.classifier[1] = nn.Sequential(
        nn.Dropout(best_params['dropout']),
        nn.Linear(model.classifier[1].in_features, 2)
    )
    model.load_state_dict(torch.load("efficientnet_final_best.pth", map_location=device))
    model.to(device).eval()
    return model

model = load_model()

# ================= TRANSFORM =================
# ✅ FIXED: Preprocessing must be normal Python code (not inside CSS / markdown)
weights = EfficientNet_B2_Weights.DEFAULT
preprocess = weights.transforms()

class_names = ["Benign", "Malignant"]
HISTORY_FILE = "patient_history.csv"

if not os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Image", "Prediction", "Confidence"])

# ================= PDF REPORT =================
def generate_pdf(filename, prediction, confidence):
    pdf_path = f"report_{filename}.pdf"
    c = canvas.Canvas(pdf_path, pagesize=letter)
    c.setFont("Helvetica-Bold", 18)
    c.drawString(200, 750, "Breast Cancer AI Report")
    c.setFont("Helvetica", 14)
    c.drawString(50, 700, f"Image: {filename}")
    c.drawString(50, 670, f"Prediction: {prediction}")
    c.drawString(50, 640, f"Confidence: {confidence:.2f}%")
    c.drawString(50, 610, f"Generated On: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    c.save()
    return pdf_path

# ================= GRAD-CAM =================
def generate_gradcam(model, image_tensor):
    gradients, activations = [], []

    def backward_hook(module, grad_in, grad_out): gradients.append(grad_out[0])
    def forward_hook(module, input, output): activations.append(output)

    target_layer = model.features[-1]
    target_layer.register_forward_hook(forward_hook)
    target_layer.register_full_backward_hook(backward_hook)

    output = model(image_tensor)
    class_score = output[0].max()
    model.zero_grad(); class_score.backward()

    grads, acts = gradients[0], activations[0]
    weights = torch.mean(grads, dim=(2,3), keepdim=True)
    cam = torch.sum(weights * acts, dim=1).squeeze()
    cam = torch.relu(cam); cam -= cam.min(); cam /= cam.max()
    # ✅ FIX: detach tensor before converting to numpy
    return cv2.resize(cam.detach().cpu().numpy(), (512,512))

# ================= DIAGNOSIS PANEL =================
if page == "Diagnosis Panel":
    st.title("🏥 AI Diagnosis Panel")
    uploaded_files = st.file_uploader("Upload Histopathology Images", type=["jpg","jpeg","png"], accept_multiple_files=True)

    if uploaded_files:
        for uploaded_file in uploaded_files:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption=uploaded_file.name)

            # ✅ FIX: Limit displayed size to prevent blur
            st.markdown("""
                <style>
                img {
                    max-height: 400px;
                    width: auto;
                }
                </style>
            """, unsafe_allow_html=True)

            img_tensor = preprocess(image).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(img_tensor)
                probs = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probs, 1)

            pred_label = class_names[predicted.item()]
            conf_percent = confidence.item() * 100

            st.markdown(f"""
<div class='card'>
<h3>🧠 AI Result</h3>
<b>Status:</b> {'🟥' if pred_label=='Malignant' else '🟩'} {pred_label}<br>
<b>Confidence:</b> {conf_percent:.2f}%
</div>
""", unsafe_allow_html=True)

            st.progress(int(conf_percent))

            with open(HISTORY_FILE, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([datetime.now(), uploaded_file.name, pred_label, conf_percent])

            cam = generate_gradcam(model, img_tensor)
            original = np.array(image.resize((512,512)))
            heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

            col1, col2 = st.columns([1,1])
            with col1: st.image(original, caption="Original")
            with col2: st.image(overlay, caption="Grad-CAM Heatmap")

            pdf_path = generate_pdf(uploaded_file.name, pred_label, conf_percent)
            with open(pdf_path, "rb") as pdf_file:
                st.download_button("📄 Download PDF Report", pdf_file, file_name=pdf_path)

# ================= PATIENT DATABASE =================
if page == "Patient Database":
    st.title("🗃️ Patient Database")
    df = pd.read_csv(HISTORY_FILE)
    st.dataframe(df, use_container_width=True)

# ================= ANALYTICS DASHBOARD =================
if page == "Analytics Dashboard":
    st.title("📈 Model Analytics Dashboard")
    df = pd.read_csv(HISTORY_FILE)

    if not df.empty:
        st.subheader("Prediction Distribution")
        st.bar_chart(df['Prediction'].value_counts())

        st.subheader("Average Confidence")
        avg_conf = df.groupby('Prediction')['Confidence'].mean()
        st.bar_chart(avg_conf)

        st.subheader("Total Cases Analyzed")
        st.metric("Total Images", len(df))

st.sidebar.markdown("---")
st.sidebar.success(f"Logged in as: {st.session_state.user}")
