import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn

# CIFAR-10 classes
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck']

# Load model
model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, 10)  # for CIFAR-10
model.load_state_dict(torch.load("resnet_cifar10.pth", map_location="cpu"))
model.eval()

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Streamlit UI
st.title("CIFAR-10 Image Classifier")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
        prediction = classes[predicted.item()]

    st.markdown(f"### Prediction: **{prediction}**")
