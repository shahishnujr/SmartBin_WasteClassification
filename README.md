# EcoSort ♻️

EcoSort is a smart waste classification system that uses object detection and deep learning to categorize waste as **biodegradable** or **non-biodegradable**. Built with Streamlit, it offers an intuitive interface for uploading images and viewing classification results with visual reports.

## 🔍 How It Works

- **YOLOv8 (Ultralytics)** detects waste items in uploaded images.
- Each detected crop is passed to a **custom PyTorch CNN** to classify it as biodegradable or non-biodegradable.
- The app generates a summary with pie charts, bar graphs, and key metrics like ratios and averages.

## 🚀 Getting Started

### 1. Clone the Repository
```
git clone https://github.com/shahishnujr/SmartBin_WasteClassification.git
cd SmartBin_WasteClassification
cd new_ml_project
```
### 2. Download Model Weights

**Note:**  
The model files `deep_custom_waste_cnn.pt`  download these files using [Google Colab]
yolov8n.pt from github

### 3. Run the App
```
streamlit run app.py
```

## 📁 File Structure
```
NEW_ML_PROJECT/
├── app.py # Main Streamlit app
├── deep_custom_waste_cnn.pt # Trained PyTorch CNN model
├── yolov8n.pt # YOLOv8 nano model
├── yolov5su.pt # (Optional/legacy) YOLOv5 model
├── crops/ # Folder to store cropped images
│ ├── crop_0_1.jpg
│ ├── crop_0_2.jpg
│ └── ...
├── .streamlit/
│ └── config.toml # Streamlit theme config (forces light mode)
├── requirements.txt # Python dependencies
├── temp_0.jpg # Temporary uploaded images
├── temp_1.jpg
├── temp_2.jpg
└── test_yolo_download.py # (Optional) Script for model download/testing
```

## 📊 Features

- Upload multiple images at once
- Detect and classify objects in trash dumps
- Track and reset classification sessions
- View real-time charts and summaries
- Lightweight UI with zero setup headaches

## 📦 Requirements

- Python 3.8+
- torch
- torchvision
- ultralytics
- opencv-python
- streamlit
- matplotlib
- Pillow

> **Note:**  
> The models (`deep_custom_waste_cnn.pt` and `yolov8n.pt`) must be present in the project root directory.









