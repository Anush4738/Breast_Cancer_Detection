# ================== IMPORTS ==================
import streamlit as st
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights
from PIL import Image
import numpy as np
import cv2
from datetime import datetime
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
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

if not st.session_state.logged_in:

    col1, col2 = st.columns(2)

    # 🔐 LOGIN
    with col1:
        st.subheader("🔐 Login")

        login_user = st.text_input("Username", key="login_user")
        login_pwd = st.text_input("Password", type="password", key="login_pwd")

        if st.button("Login"):
            users = db.collection("users").stream()

            for u in users:
                data = u.to_dict()
                if data["username"] == login_user and data["password"] == login_pwd:
                    st.session_state.logged_in = True
                    st.session_state.user = login_user
                    st.rerun()

            st.error("Invalid credentials ❌")

    # 🆕 SIGNUP
    with col2:
        st.subheader("🆕 Signup")

        signup_user = st.text_input("New Username", key="signup_user")
        signup_pwd = st.text_input("New Password", type="password", key="signup_pwd")

        if st.button("Create Account"):
            if signup_user and signup_pwd:
                db.collection("users").add({
                    "username": signup_user,
                    "password": signup_pwd
                })
                st.success("Account Created ✅")
            else:
                st.warning("Enter all details")

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
def generate_pdf(patient_name, age, gender, prediction, confidence):
    path = f"{patient_name}_report.pdf"
    doc = SimpleDocTemplate(path)
    styles = getSampleStyleSheet()

    content = []
    content.append(Paragraph("<b>Breast Cancer Report</b>", styles["Title"]))
    content.append(Spacer(1, 20))

    content.append(Paragraph(f"<b>Patient Name:</b> {patient_name}", styles["Normal"]))
    content.append(Paragraph(f"<b>Age:</b> {age}", styles["Normal"]))
    content.append(Paragraph(f"<b>Gender:</b> {gender}", styles["Normal"]))
    content.append(Spacer(1, 20))

    content.append(Paragraph(f"<b>Prediction:</b> {prediction}", styles["Normal"]))
    content.append(Paragraph(f"<b>Confidence:</b> {confidence:.2f}%", styles["Normal"]))

    doc.build(content)
    return path

# ================= DIAGNOSIS =================
if page == "Diagnosis Panel":

    st.subheader("🧑 Patient Details")

    patient_name = st.text_input("Patient Name")
    patient_age = st.number_input("Age", 0, 120)
    patient_gender = st.selectbox("Gender", ["Male", "Female", "Other"])

    uploaded_files = st.file_uploader("Upload Images", type=["jpg","png"], accept_multiple_files=True)

    if uploaded_files:
        for file in uploaded_files:

            if not patient_name:
                st.warning("Enter patient name first ⚠️")
                st.stop()

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

            # GradCAM
            cam = generate_gradcam(model, tensor)
            original = np.array(image.resize((512, 512)))
            heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

            col1, col2 = st.columns(2)
            col1.image(original, caption="Original")
            col2.image(overlay, caption="Grad-CAM")

            # Save to Firebase
            db.collection("patients").add({
                "patient_name": patient_name,
                "age": patient_age,
                "gender": patient_gender,
                "prediction": label,
                "confidence": conf,
                "image": file.name,
                "user": st.session_state.user,
                "timestamp": str(datetime.now())
            })

            st.success("Saved to database ✅")

            pdf = generate_pdf(patient_name, patient_age, patient_gender, label, conf)

            with open(pdf, "rb") as f:
                st.download_button("Download Report", f, file_name=pdf)

# ================= DATABASE =================
if page == "Patient Database":

    docs = db.collection("patients").stream()

    data = [d.to_dict() for d in docs]

    if data:
        df = pd.DataFrame(data)

        st.dataframe(df[[
            "patient_name",
            "age",
            "gender",
            "prediction",
            "confidence",
            "user",
            "timestamp"
        ]])
    else:
        st.warning("No data yet")

# ================= DASHBOARD =================
if page == "Analytics Dashboard":

    docs = db.collection("patients").stream()
    data = [d.to_dict() for d in docs]

    if data:
        df = pd.DataFrame(data)

        st.bar_chart(df["prediction"].value_counts())
        st.metric("Total Cases", len(df))