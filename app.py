# ================== IMPORTS ==================
import streamlit as st
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights
from PIL import Image
import numpy as np
import cv2
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import pandas as pd

# 🔥 FIREBASE
import firebase_admin
from firebase_admin import credentials, firestore

# ================== FIREBASE INIT ==================
if not firebase_admin._apps:
    firebase_dict = dict(st.secrets["firebase"])
    firebase_dict["private_key"] = firebase_dict["private_key"].replace("\\n", "\n")

    cred = credentials.Certificate(firebase_dict)
    firebase_admin.initialize_app(cred)

db = firestore.client()

# ================== CONFIG ==================
st.set_page_config(page_title="AI Breast Cancer Detection", layout="wide")

# ================== HEADER ==================
st.markdown("""
<h1 style='text-align: center; color: #0b3d91;'>🧬 AI Breast Cancer Detection</h1>
<p style='text-align: center; color: gray;'>Clinical AI for Histopathology Analysis</p>
<hr>
""", unsafe_allow_html=True)

# ================== SESSION ==================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# ================== AUTH ==================
menu = st.selectbox("Select Option", ["Login", "Signup"])

if not st.session_state.logged_in:

    if menu == "Signup":
        st.subheader("Signup")
        user = st.text_input("Username")
        pwd = st.text_input("Password", type="password")

        if st.button("Create Account"):
            db.collection("users").add({
                "username": user,
                "password": pwd
            })
            st.success("Account Created ✅")

    if menu == "Login":
        st.subheader("Login")
        user = st.text_input("Username")
        pwd = st.text_input("Password", type="password")

        if st.button("Login"):
            users = db.collection("users").stream()

            for u in users:
                data = u.to_dict()
                if data["username"] == user and data["password"] == pwd:
                    st.session_state.logged_in = True
                    st.session_state.user = user
                    st.rerun()

            st.error("Invalid credentials ❌")

    st.stop()

# ================== SIDEBAR ==================
st.sidebar.markdown("## 🏥 AI Panel")
st.sidebar.info(f"👤 {st.session_state.user}")

if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.rerun()

page = st.sidebar.radio("Navigate", [
    "Diagnosis Panel",
    "Patient Database",
    "Analytics Dashboard"
])

# ================= MODEL =================
device = torch.device("cpu")

@st.cache_resource
def load_model():
    model = efficientnet_b2(weights=EfficientNet_B2_Weights.DEFAULT)

    model.classifier[1] = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.classifier[1].in_features, 2)
    )

    state_dict = torch.load("efficientnet_final_best.pth", map_location=device)
    model.load_state_dict(state_dict, strict=False)

    model.eval()
    return model

model = load_model()
preprocess = EfficientNet_B2_Weights.DEFAULT.transforms()
class_names = ["Benign", "Malignant"]

# ================= GRADCAM =================
def generate_gradcam(model, image_tensor):
    gradients = []
    activations = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    target_layer = model.features[-1]

    fwd = target_layer.register_forward_hook(forward_hook)
    bwd = target_layer.register_full_backward_hook(backward_hook)

    output = model(image_tensor)
    loss = output[0].max()

    model.zero_grad()
    loss.backward()

    grads = gradients[0]
    acts = activations[0]

    weights = torch.mean(grads, dim=(2, 3), keepdim=True)
    cam = torch.sum(weights * acts, dim=1).squeeze()

    cam = torch.relu(cam)
    cam -= cam.min()
    cam /= cam.max()

    fwd.remove()
    bwd.remove()

    return cv2.resize(cam.detach().numpy(), (512, 512))

# ================= PDF =================
def generate_pdf(filename, prediction, confidence):
    path = f"report_{filename}.pdf"
    c = canvas.Canvas(path, pagesize=letter)
    c.drawString(100, 750, "Breast Cancer Report")
    c.drawString(100, 700, f"Prediction: {prediction}")
    c.drawString(100, 670, f"Confidence: {confidence:.2f}%")
    c.save()
    return path

# ================= DIAGNOSIS =================
if page == "Diagnosis Panel":

    uploaded_files = st.file_uploader("Upload Images", type=["jpg","png"], accept_multiple_files=True)

    if uploaded_files:
        for file in uploaded_files:
            image = Image.open(file).convert("RGB")
            st.image(image)

            tensor = preprocess(image).unsqueeze(0)

            with torch.no_grad():
                out = model(tensor)
                prob = torch.softmax(out, dim=1)
                conf, pred = torch.max(prob,1)

            label = class_names[pred.item()]
            conf = conf.item()*100

            st.success(f"{label} ({conf:.2f}%)")

            # 🔥 GRADCAM
            cam = generate_gradcam(model, tensor)
            original = np.array(image.resize((512, 512)))
            heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

            col1, col2 = st.columns(2)
            with col1:
                st.image(original, caption="Original")
            with col2:
                st.image(overlay, caption="AI Focus (Grad-CAM)")

            # 🔥 SAVE TO FIREBASE
            db.collection("patients").add({
                "timestamp": str(datetime.now()),
                "image": file.name,
                "prediction": label,
                "confidence": conf,
                "user": st.session_state.user
            })

            st.success("Saved to database ✅")

            pdf = generate_pdf(file.name, label, conf)

            with open(pdf, "rb") as f:
                st.download_button("Download Report", f, file_name=pdf)

# ================= DATABASE =================
if page == "Patient Database":

    docs = db.collection("patients").stream()

    data = []
    for d in docs:
        data.append(d.to_dict())

    if data:
        df = pd.DataFrame(data)
        st.dataframe(df)
    else:
        st.warning("No data yet")

# ================= DASHBOARD =================
if page == "Analytics Dashboard":

    docs = db.collection("patients").stream()

    data = []
    for d in docs:
        data.append(d.to_dict())

    if data:
        df = pd.DataFrame(data)

        st.bar_chart(df["prediction"].value_counts())
        st.metric("Total Cases", len(df))