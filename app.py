# ================== IMPORTS ==================
import streamlit as st
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights
from PIL import Image
import numpy as np
import cv2
import os
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

    # 🔥 FIX PRIVATE KEY ISSUE
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
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    model.load_state_dict(torch.load("efficientnet_final_best.pth", map_location=device))
    model.eval()
    return model

model = load_model()
preprocess = EfficientNet_B2_Weights.DEFAULT.transforms()
class_names = ["Benign", "Malignant"]

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