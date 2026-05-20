import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import streamlit as st
import matplotlib.cm as cm
from torchvision import transforms
from transformers import AutoModelForImageClassification
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

st.set_page_config(page_title="Radiologie médicale", page_icon="🩻", layout="wide")

MODEL_NAME = "google/vit-base-patch16-224"
CLASS_NAMES = {0: "NORMAL", 1: "PNEUMONIA"}

@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForImageClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        ignore_mismatched_sizes=True,
    ).to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = True
    return model, device

model, device = load_model()

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result

class HFGradCAMWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(pixel_values=x).logits

@st.cache_resource
def load_cam(_model):
    wrapped_model = HFGradCAMWrapper(_model).to(device).eval()
    target_layer = _model.vit.encoder.layer[-2].layernorm_before
    cam = GradCAM(
        model=wrapped_model,
        target_layers=[target_layer],
        reshape_transform=reshape_transform
    )
    return cam

cam = load_cam(model)

def denormalize(img_tensor):
    img_np = img_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = img_np * std + mean
    return np.clip(img_np, 0, 1)

st.title("🩻 Radiologie médicale")
st.write(
    "Application de démonstration pour l'analyse de radios thoraciques avec "
    "**prédiction**, **score de confiance** et **visualisation GradCAM**."
)

uploaded_file = st.file_uploader(
    "Téléversez une image radiologique",
    type=["png", "jpg", "jpeg"]
)

st.warning(
    "**Disclaimer médical :** cette application est un outil d'aide à la décision uniquement. "
    "Elle ne remplace pas l'avis d'un radiologue ou d'un médecin."
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(image, caption="Image uploadée", use_container_width=True)

    with st.spinner("Analyse en cours...", show_time=True):
        input_tensor = val_transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(pixel_values=input_tensor).logits
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()[0]
            pred_idx = int(np.argmax(probs))
            confidence = float(probs[pred_idx])

        normal_prob = float(probs[0])
        pneumonia_prob = float(probs[1])

        grayscale_cam = cam(
            input_tensor=input_tensor,
            targets=[ClassifierOutputTarget(pred_idx)]
        )[0]

        original_rgb = denormalize(input_tensor[0])
        overlay = show_cam_on_image(original_rgb, grayscale_cam, use_rgb=True)
        heatmap = (cm.jet(grayscale_cam)[..., :3] * 255).astype(np.uint8)

    with col2:
        st.subheader("Résultat de la prédiction")
        st.success(f"Prédiction : {CLASS_NAMES[pred_idx]}")
        st.info(f"Score de confiance : {confidence:.3f}")
        st.markdown("### Probabilités détaillées")
        st.write(f"NORMAL : {normal_prob:.3f}")
        st.write(f"PNEUMONIA : {pneumonia_prob:.3f}")

    st.subheader("Visualisation GradCAM")
    col3, col4 = st.columns(2)

    with col3:
        st.image(heatmap, caption="Heatmap GradCAM", use_container_width=True)

    with col4:
        st.image(overlay, caption="Overlay GradCAM", use_container_width=True)
