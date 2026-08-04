import base64
import io
import cv2
import numpy as np
import streamlit as st
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import torch.nn as nn
from torchvision import models

st.set_page_config(
    page_title="PCB Micro-Defect Inspection System",
    page_icon="🔍",
    layout="wide"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_pcb_model():
    weights = models.EfficientNet_B2_Weights.DEFAULT
    model = models.efficientnet_b2(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 1)
    )
    return model

@st.cache_resource
def load_model():
    model = get_pcb_model().to(device)
    MODEL_PATH = "pcb_defect_efficientnet_focal.pth"
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model

model = load_model()
target_layer = model.features[-1]

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        
        def forward_hook(module, input, output):
            self.activations = output
            
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
            
        
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

    def generate_heatmap(self, input_tensor):
        
        output = self.model(input_tensor)
        
        
        self.model.zero_grad()
        output.backward()
        
        
        gradients = self.gradients.detach().cpu().numpy()[0]
        activations = self.activations.detach().cpu().numpy()[0]
        
    
        weights = np.mean(gradients, axis=(1, 2))
        
        
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]
        
        
        cam = np.maximum(cam, 0)
        
        
        cam = cv2.resize(cam, (224, 224))
        
        
        if np.max(cam) > 0:
            cam = cam / np.max(cam)
        
        
        confidence = torch.sigmoid(output).item()
        
        return cam, confidence

grad_cam = GradCAM(model, target_layer)


val_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

def predict_pcb(image_np, threshold=0.75):
    orig_h, orig_w, _ = image_np.shape
    
    
    augmented = val_transform(image=image_np)
    input_tensor = augmented['image'].unsqueeze(0).to(device)
    
    
    heatmap, confidence = grad_cam.generate_heatmap(input_tensor)
    
    
    heatmap_resized = cv2.resize(heatmap, (orig_w, orig_h))
    
    
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(image_np, 0.6, heatmap_colored, 0.4, 0)
    
    
    overlay_pil = Image.fromarray(overlay)
    buffered = io.BytesIO()
    overlay_pil.save(buffered, format="PNG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    is_defective = confidence >= threshold
    pred_label = "Defective" if is_defective else "Clean / Normal"
    
    return {
        "prediction": pred_label,
        "defect_probability": float(confidence),
        "overlay_base64": img_b64,
        "xai_explanation": {
            "RED": "High Focus Region (Potential Anomaly / Key Feature)",
            "YELLOW_GREEN": "Moderate Feature Check Area",
            "BLUE": "Clean / Neutral Background Area"
        }
    }


st.title("🏭 Automated PCB Micro-Defect Quality Assurance")
st.markdown("Upload any image to run Deep Learning Diagnosis with Explainable Heatmaps.")


st.sidebar.header("⚙️ Inspection Controls")
threshold = st.sidebar.slider(
    "Defect Detection Threshold", 
    min_value=0.50, 
    max_value=0.95, 
    value=0.75, 
    step=0.05,
    help="Higher threshold reduces false alarms."
)


uploaded_file = st.file_uploader("Choose an Image...", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    
    
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    
    with col1:
        st.subheader("📷 Original Uploaded Image")
        st.image(image, use_container_width=True)
    
    
    if st.button("🔎 Inspect Image"):
        with st.spinner("Processing image & generating Grad-CAM heatmap..."):
            try:
                
                data = predict_pcb(image_np, threshold)
                
                
                prediction = data["prediction"]
                prob = data["defect_probability"]
                overlay_b64 = data["overlay_base64"]
                explanations = data["xai_explanation"]
                
                
                overlay_img = Image.open(io.BytesIO(base64.b64decode(overlay_b64)))
                with col2:
                    st.subheader("🔥 Grad-CAM Defect Heatmap")
                    st.image(overlay_img, use_container_width=True)
                
                st.divider()
                
                
                if "Defective" in prediction:
                    st.error(f"⚠️ **Result: DEFECTIVE** (Defect Score: {prob*100:.1f}% | Threshold: {threshold*100:.0f}%)")
                else:
                    st.success(f"✅ **Result: CLEAN / NORMAL** (Defect Score: {prob*100:.1f}% | Threshold: {threshold*100:.0f}%)")
                
                
                st.subheader("💡 Explainable AI (Grad-CAM) Visual Guide")
                exp_col1, exp_col2, exp_col3 = st.columns(3)
                
                with exp_col1:
                    st.info(f"🔴 **Red Regions**\n\n{explanations['RED']}")
                with exp_col2:
                    st.warning(f"🟡🟢 **Yellow/Green**\n\n{explanations['YELLOW_GREEN']}")
                with exp_col3:
                    st.success(f"🔵 **Blue Regions**\n\n{explanations['BLUE']}")
                    
            except Exception as e:
                st.error(f"Error during inference: {e}")
                st.code(f"Traceback: {str(e)}")