# run this command in terminal : streamlit run app.py
import streamlit as st
import os
import shutil
import torch
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
from ultralytics import YOLO

st.set_page_config(page_title="EcoSort", layout="wide")

st.markdown("""
    <style>
    html, body, [class*="css"] {
        font-family: 'Segoe UI', sans-serif;
        background-color: #f4f6f9;
    }
    .title-style {
        background: linear-gradient(90deg, #43cea2 0%, #185a9d 100%);
        color: white;
        text-align: center;
        padding: 1rem;
        font-size: 28px;
        font-weight: 700;
        border-radius: 12px;
        margin-top: 1rem;
        margin-bottom: 2rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    .stButton>button {
        background: linear-gradient(to right, #00b09b, #96c93d);
        color: white;
        font-weight: 600;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        border: none;
    }
    .report-box {
        background-color: #1e1e1e;
        color: white;
        border-radius: 10px;
        padding: 1rem;
        font-size: 16px;
        box-shadow: 0 0 10px rgba(0,0,0,0.05);
        margin-top: 1rem;
    }
    .section-title {
        font-size: 20px;
        margin-top: 2rem;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

IMG_SIZE = 128
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

class DeepWasteCNN(nn.Module):
    def __init__(self):
        super(DeepWasteCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.25),
            nn.Conv2d(32, 64, kernel_size=3), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.25)
        )
        self._to_linear = None
        self._get_flattened_size()
        self.fc = nn.Sequential(
            nn.Linear(self._to_linear, 512), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2)
        )

    def _get_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)
            x = self.conv(x)
            self._to_linear = x.numel()

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

@st.cache_resource
def load_models():
    cnn_model = DeepWasteCNN()
    cnn_model.load_state_dict(torch.load("deep_custom_waste_cnn.pt", map_location='cpu'))
    cnn_model.eval()
    yolo_model = YOLO("yolov8n.pt")
    return cnn_model, yolo_model

st.info("🚀 Initializing models...")
cnn_model, yolo_model = load_models()
st.success("✅ Models loaded successfully!")

crop_dir = "crops"

def clear_all():
    if os.path.exists(crop_dir):
        shutil.rmtree(crop_dir)
    os.makedirs(crop_dir, exist_ok=True)
    st.session_state['bio'] = 0
    st.session_state['nonbio'] = 0
    st.session_state['dumps'] = 0
    st.session_state['images'] = []

def classify_crop(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = cnn_model(image)
        _, pred = torch.max(output, 1)
        return "biodegradable" if pred.item() == 0 else "non_biodegradable"

if 'bio' not in st.session_state:
    clear_all()

st.markdown('<div class="title-style">EcoSort – Smart Bin Waste Classifier</div>', unsafe_allow_html=True)
st.markdown("Upload images of trash dumps to automatically detect and classify **biodegradable** and **non-biodegradable** waste.")

st.sidebar.title("🧰 Tools")
if st.sidebar.button("🧹 Reset - Empty Bin"):
    clear_all()
    st.success("Session reset successfully!")

uploaded_files = st.sidebar.file_uploader("📤 Upload Images", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    for i, file in enumerate(uploaded_files):
        img = Image.open(file).convert("RGB")
        st.session_state['images'].append(img)
        img_path = f"temp_{i}.jpg"
        img.save(img_path)

        results = yolo_model(img_path)
        st.image(img, caption=f"🧾 Dump #{i+1}", width=250)

        if results[0].boxes is None or len(results[0].boxes) == 0:
            st.warning("⚠ No objects detected. Classifying entire image.")
            label = classify_crop(img_path)
            st.session_state['bio' if label == "biodegradable" else 'nonbio'] += 1
        else:
            boxes = results[0].boxes
            names = results[0].names
            image_cv2 = cv2.imread(img_path)
            shown_classes = set()

            for j, box in enumerate(boxes):
                cls_id = int(box.cls[0])
                class_name = names[cls_id].lower()

                if class_name == "person":
                    continue

                if class_name == "banana":
                    label = "biodegradable"
            
                else:
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = xyxy
                    crop = image_cv2[y1:y2, x1:x2]
                    crop_path = os.path.join(crop_dir, f"crop_{i}_{j}.jpg")
                    cv2.imwrite(crop_path, crop)
                    label = classify_crop(crop_path)

                st.session_state['bio' if label == "biodegradable" else 'nonbio'] += 1

                if class_name not in shown_classes:
                    st.write(f"🟢 `{class_name}` → **{label}**")
                    shown_classes.add(class_name)

        st.session_state['dumps'] += 1
        st.success(f"✅ Processed Dump #{st.session_state['dumps']}")

total = st.session_state['bio'] + st.session_state['nonbio']
if total > 0:
    st.markdown('<div class="section-title">📊 Waste Classification Summary</div>', unsafe_allow_html=True)

    labels = ['Biodegradable', 'Non-Biodegradable']
    sizes = [st.session_state['bio'], st.session_state['nonbio']]
    colors = ['#00C853', '#FF6F00']

    col1, col2 = st.columns(2)
    with col1:
        fig1, ax1 = plt.subplots()
        ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax1.axis('equal')
        st.pyplot(fig1)

    with col2:
        fig2, ax2 = plt.subplots()
        ax2.bar(labels, sizes, color=colors)
        ax2.set_ylabel("Items")
        st.pyplot(fig2)

    avg_items = round(total / st.session_state['dumps'], 2)
    ratio = round(st.session_state['bio'] / (st.session_state['nonbio'] + 1e-5), 2)

    st.markdown(f"""
    <div class="report-box">
    <b>Total Items:</b> {total}<br>
    ✅ <b>Biodegradable:</b> {st.session_state['bio']}<br>
    ❌ <b>Non-Biodegradable:</b> {st.session_state['nonbio']}<br>
    📦 <b>Average per Dump:</b> {avg_items}<br>
    ♻ <b>Bio : Non-Bio Ratio:</b> {ratio} : 1
    </div>
    """, unsafe_allow_html=True)

    if sizes[0] > sizes[1]:
        st.success("🌱 More biodegradable waste detected – great for composting.")
    elif sizes[1] > sizes[0]:
        st.warning("🧴 More non-biodegradable items – reduce plastic usage.")
    else:
        st.info("⚖ Balanced waste detected – good effort!")

if st.session_state['images']:
    st.markdown('<div class="section-title">🖼 All Uploaded Dumps</div>', unsafe_allow_html=True)
    st.image(st.session_state['images'], width=150)

st.markdown("""
---
<center>
    <sub>Powered by <b>Streamlit</b>, <b>YOLOv8</b>, and <b>PyTorch CNN</b><br>
    <i>EcoSort – Waste Classification Assistant</i>
</sub>
</center>
""", unsafe_allow_html=True)
