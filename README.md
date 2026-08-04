# PCB Defect Inspection

A practical deep-learning project for **automatic PCB defect detection and inspection** using image-based analysis. This repository focuses on identifying manufacturing defects on printed circuit boards (PCBs) to support faster, more reliable quality control.

---

## 📌 Table of Contents
- [Project Overview](#-project-overview)
- [Real-World Problem](#-real-world-problem)
- [My Solution](#-my-solution)
- [How the Project Works](#-how-the-project-works)
- [Tech Stack](#-tech-stack)
- [How to Run](#-how-to-run)
- [Streamlit Live Demo](#-streamlit-live-demo)
---

## 🚀 Project Overview
**PCB Defect Inspection** is an AI-powered inspection workflow that analyzes PCB images and helps detect defects automatically.

Instead of fully manual visual checks, this project demonstrates how computer vision and deep learning can:
- reduce inspection time,
- improve consistency,
- and support scalable production quality monitoring.

---

## 🌍 Real-World Problem
In electronics manufacturing, PCB quality is critical. Even small defects can cause:
- product failure,
- higher return rates,
- increased manufacturing cost,
- and safety/reliability issues.

Traditional manual inspection is often:
- time-consuming,
- operator-dependent,
- and difficult to scale in high-volume production.

---

## 💡 My Solution
This project provides a machine learning–based pipeline that:
1. takes PCB image inputs,
2. processes and analyzes them using a trained model,
3. predicts whether defects are present,
4. and supports decision-making for quality control.

The goal is to assist engineers and manufacturers with **faster and more objective defect identification**.

---

## ⚙️ How the Project Works
At a high level, the workflow is:

1. **Data Input**  
   PCB images are loaded from the dataset.

2. **Preprocessing**  
   Images are cleaned/resized/normalized for model compatibility.

3. **Model Inference / Training**  
   A deep learning model is trained (or loaded) to learn defect patterns.

4. **Prediction**  
   The model outputs defect-related predictions from test images.

5. **Evaluation & Analysis**  
   Results are reviewed to understand performance and reliability.

> The exact implementation details are available in the project notebooks and scripts.

---

## 🧰 Tech Stack
- **Language:** Python
- **Development Format:** Jupyter Notebook
- **Core Domain:** Computer Vision / Deep Learning(Covolutional Neural Network)
- **Potential Libraries Used:** NumPy, Pandas, Matplotlib, OpenCV, PyTorch,EfficientNet-B2(Transfer Learning)

---



## ▶️ How to Run
### 1) Clone the repository
```bash
git clone https://github.com/Furqan-Ahmed2006/PCB-Defect-Inspection.git
cd PCB-Defect-Inspection
```

### 2) Create and activate a virtual environment (recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3) Install dependencies
```bash
pip install -r requirements.txt
```

### 4) Run notebooks / scripts
- Open Jupyter Notebook or JupyterLab and run the project notebook(s):
```bash
jupyter notebook
```
- Or run Python scripts directly 

---

## 🌐 Streamlit Live Demo


**🔗 Demo Link:** https://pcb-defect-inspection-tbzwkxnxkcwp7usawqhpkz.streamlit.app/


---

