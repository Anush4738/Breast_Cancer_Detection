# 🩺 Breast Cancer Detection Using Deep Learning

## 📌 Project Title

**Comparative Deep Learning Study for Breast Cancer Histopathology Image Classification**  
MobileNetV2 vs EfficientNet-B2 with Advanced Optimization Techniques

---

## 🌐 Live Web App

👉 **Try the deployed application here:**  
🔗 [https://breastcancerdetection-b6ejkc7irq3cnzbcjt837g.streamlit.app](https://breastcancerdetection-mrbbtpzpu6g5xtd4dpqyql.streamlit.app/)

> ⚠️ Note: You can create a new account using Signup and then Login to use the system.

---

## 📖 Project Overview

This project presents an AI-based system for automated classification of breast cancer histopathology images using deep learning. The primary goal is to assist pathologists by providing fast, accurate, and consistent diagnostic predictions, reducing human error and mitigating challenges such as inter-observer variability and time constraints.

Two deep learning architectures were implemented and compared:

* ✅ **MobileNetV2** – Baseline lightweight model for efficient performance  
* 🚀 **EfficientNet-B2** – Advanced model enhanced with MixUp augmentation and Optuna hyperparameter optimization  

The system demonstrates how architectural improvements combined with advanced optimization techniques significantly enhance classification accuracy and clinical reliability.

---

## 🎯 Objectives

* Develop a baseline breast cancer classification model using MobileNetV2  
* Implement an advanced model using EfficientNet-B2  
* Address dataset class imbalance using weighted sampling and class-weighted loss  
* Apply MixUp augmentation to improve generalization  
* Optimize hyperparameters systematically using Optuna  
* Compare performance using multiple evaluation metrics  

---

## 🧠 Methodology

### Workflow Overview

1. Merge two large histopathology datasets  
2. Apply preprocessing and augmentation  
3. Split data into Train / Validation / Test sets (70/15/15)  
4. Train baseline MobileNetV2  
5. Train EfficientNet-B2 with MixUp + Optuna  
6. Evaluate and compare results  

### Data Sources

* Kaggle Breast Histopathology Dataset – 277,524 images  
* BreaKHis Dataset – 7,909 images  
* Total Samples: **285,433 images**  
* Image Size: 224×224 pixels  
* Format: RGB  

### Techniques Used

* PyTorch Framework  
* WeightedRandomSampler  
* Class-weighted CrossEntropy Loss  
* MixUp Data Augmentation  
* Grad-CAM Visualization  
* Optuna Hyperparameter Optimization  

---

## 📊 Performance Comparison

| Model           | Validation Accuracy | Recall (Minority Class) | False Negative Rate |
|----------------|-------------------|------------------------|---------------------|
| MobileNetV2     | 87.78%            | 85.91%                 | 14.09%              |
| EfficientNet-B2 | **92.15%**        | **91.28%**             | **8.72%**           |

✅ EfficientNet-B2 outperformed MobileNetV2 in all critical metrics, making it more suitable for clinical decision support systems.

---

## 🏥 Clinical Significance

* Higher sensitivity reduces chances of missed cancer cases  
* AI supports pathologists with faster preliminary diagnosis  
* Improves consistency and reliability in medical imaging evaluation  
* Enhances early detection effectiveness  

---

## 🖥️ Web Application Features

* 🔐 Secure Login & Signup System (Firebase-based authentication)  
* 🧑 Patient Information Input (Name, Age, Gender)  
* 🧠 AI Diagnosis Panel (Upload Histopathology Images)  
* ⚡ Real-time Prediction with Confidence Score  
* 🔥 Grad-CAM Heatmap Visualization (Explainable AI)  
* 🧾 PDF Report Generation with Patient Details  
* 🗂️ Patient Database (stored in Firebase Firestore)  
* 📊 Analytics Dashboard (prediction insights)  
* ☁️ Fully deployed on Streamlit Cloud  

---

## ⚙️ Technologies Used

* Python  
* PyTorch  
* Streamlit  
* Firebase (Firestore Database)  
* OpenCV  
* EfficientNet-B2  
* MobileNetV2  
* Optuna  
* ReportLab  
* NumPy & Pandas  

---

## 🚀 Installation & Setup

```bash
git clone https://github.com/Anush4738/Breast_Cancer_Detection.git
cd Breast_Cancer_Detection
pip install -r requirements.txt
streamlit run app.py
