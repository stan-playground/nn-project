import streamlit as st
import torch
from PIL import Image
from model.model import Classifier
from model.preprocessing import preprocess

import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

import requests
from io import BytesIO
import time

class Myresnet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.model.fc = nn.Linear(512, 11)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.fc.weight.requires_grad = True
        self.model.fc.bias.requires_grad = True

    def forward(self, x):
        x = self.model(x)
        return x

@st.cache_resource
def load_model():
    model = Myresnet18()
    model.eval()
    return model

model = load_model()

def load_image_from_url(url):
    response = requests.get(url)
    if response.status_code == 200:
        image = Image.open(BytesIO(response.content))
        return image
    else:
        st.error("Failed to load image from URL")
        return None
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

st.title('ResNet Image Classification')

# URL input for image
image_url = st.text_input("Enter image URL:")

if image_url:
    image = load_image_from_url(image_url)

    if image:
        st.image(image, caption='Uploaded Image.', use_column_width=True)

        # Preprocess the image
        preprocess = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            #T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image_preprocessed = preprocess(image).unsqueeze(0).to(device)

        start_time = time.time()
        with torch.no_grad():
            pred = model(image_preprocessed)
            pred_class = pred.argmax(dim=1).item()
        end_time = time.time()
        prediction_time = end_time - start_time

        confidence = torch.nn.functional.softmax(pred, dim=1)[0][pred_class].item()

        st.write(f"Class: {pred_class}")
        st.write(f"Confidence: {confidence:.2f}")
        st.write(f"Prediction time: {prediction_time:.4f} seconds")