import streamlit as st
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights
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

# ================== SIGNUP + LOGIN SYSTEM ==================
USERS_FILE = "users.csv"

# create file if not exists
if not os.path.exists(USERS_FILE):
    with open(USERS_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["username", "password"])

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

menu = st.selectbox("Select Option", ["Login", "Signup"])

if not st.session_state.logged_in:

    if menu == "Signup":
        st.title("📝 Signup")

        new_user = st.text_input("Create Username")
        new_pass = st.text_input("Create Password", type="password")

        if st.button("Signup"):
            try:
                df = pd.read_csv(USERS_FILE)
            except pd.errors.EmptyDataError:
                df = pd.DataFrame(columns=["username", "password"])

            if new_user in df["username"].values:
                st.error("User already exists ❌")
            else:
                with open(USERS_FILE, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([new_user, new_pass])

                st.success("Account created successfully ✅")

    elif menu == "Login":
        st.title("🔐 Login")

        username = st.text_input("Username").strip()
        password = st.text_input("Password", type="password").strip()

        if st.button("Login"):
            try:
                df = pd.read_csv(USERS_FILE)
            except pd.errors.EmptyDataError:
                df = pd.DataFrame(columns=["username", "password"])

            user_row = df[df["username"] == username]

            if not user_row.empty:
                if user_row.iloc[0]["password"] == password:
                    st.session_state.logged_in = True
                    st.session_state.user = username
                    st.success("Login successful ✅")
                    st.rerun()
                else:
                    st.error("Wrong password ❌")
            else:
                st.error("User not found ❌")

    st.stop()

# ================== SIDEBAR ==================
st.sidebar.title("🏥 Hospital AI Panel")
st.sidebar.write(f"👤 {st.session_state.user}")

page = st.sidebar.radio("Navigate", [
    "Diagnosis Panel",
    "Patient Database",
    "Analytics Dashboard"
])

# ================= DEVICE =================
device = torch.device("cpu")

# ================= MODEL =================
@st.cache_resource
def load_model():
    weights = EfficientNet_B2_Weights.DEFAULT
    model = efficientnet_b2(weights=weights)

    model.classifier[1] = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.classifier[1].in_features, 2)
    )

    model.load_state_dict(torch.load("efficientnet_final_best.pth", map_location=device))
    model.eval()
    return model

model = load_model()

# ================= PREPROCESS =================
weights = EfficientNet_B2_Weights.DEFAULT
preprocess = weights.transforms()

class_names = ["Benign", "Malignant"]

# ================= CSV =================
HISTORY_FILE = "patient_history.csv"

if not os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Image", "Prediction", "Confidence"])

# ================= PDF =================
def generate_pdf(filename, prediction, confidence):
    pdf_path = f"report_{filename}.pdf"
    c = canvas.Canvas(pdf_path, pagesize=letter)

    c.drawString(100, 750, "Breast Cancer Report")
    c.drawString(100, 700, f"Image: {filename}")
    c.drawString(100, 670, f"Prediction: {prediction}")
    c.drawString(100, 640, f"Confidence: {confidence:.2f}%")

    c.save()
    return pdf_path

# ================= GRADCAM =================
def generate_gradcam(model, image_tensor):
    gradients = []
    activations = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    target_layer = model.features[-1]

    f_handle = target_layer.register_forward_hook(forward_hook)
    b_handle = target_layer.register_full_backward_hook(backward_hook)

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

    f_handle.remove()
    b_handle.remove()

    return cv2.resize(cam.detach().numpy(), (512, 512))

# ================= DIAGNOSIS =================
if page == "Diagnosis Panel":
    st.title("🧬 AI Diagnosis Panel")

    uploaded_files = st.file_uploader(
        "Upload Images",
        type=["jpg", "png", "jpeg"],
        accept_multiple_files=True
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption=uploaded_file.name)

            img_tensor = preprocess(image).unsqueeze(0)

            with torch.no_grad():
                outputs = model(img_tensor)
                probs = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probs, 1)

            label = class_names[predicted.item()]
            conf = confidence.item() * 100

            st.success(f"{label} ({conf:.2f}%)")
            st.progress(int(conf))

            with open(HISTORY_FILE, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([datetime.now(), uploaded_file.name, label, conf])

            cam = generate_gradcam(model, img_tensor)
            original = np.array(image.resize((512, 512)))

            heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

            col1, col2 = st.columns(2)
            with col1:
                st.image(original, caption="Original")
            with col2:
                st.image(overlay, caption="Grad-CAM")

            pdf = generate_pdf(uploaded_file.name, label, conf)

            with open(pdf, "rb") as f:
                st.download_button("Download Report", f, file_name=pdf)

# ================= DATABASE =================
if page == "Patient Database":
    st.title("📂 Patient Records")

    if os.path.exists(HISTORY_FILE):
        df = pd.read_csv(HISTORY_FILE)
        st.dataframe(df)
    else:
        st.warning("No records yet")

# ================= DASHBOARD =================
if page == "Analytics Dashboard":
    st.title("📊 Dashboard")

    if os.path.exists(HISTORY_FILE):
        df = pd.read_csv(HISTORY_FILE)

        if not df.empty:
            st.bar_chart(df["Prediction"].value_counts())

            avg = df.groupby("Prediction")["Confidence"].mean()
            st.bar_chart(avg)

            st.metric("Total Cases", len(df))
        else:
            st.info("No data yet")
    else:
        st.warning("No data file")

st.sidebar.success(f"Logged in as: {st.session_state.user}")