import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from PIL import Image
import requests
from io import BytesIO
import time

@st.cache_resource
def load_model():
    class Myresnet18(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
            self.model.fc = nn.Linear(512, 11)
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.fc.weight.requires_grad = True
            self.model.fc.bias.requires_grad = True
        
        def forward(self, x):
            return self.model(x)

    model = Myresnet18()
    return model

def load_image_from_url(url):
    response = requests.get(url)
    return Image.open(BytesIO(response.content))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model().to(device)

st.title('Классификация ResNet')
image_url = st.text_input("Ссылка на изображение")

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

if image_url:
    try:
        image = load_image_from_url(image_url)
        st.image(image, caption='Ваше изображение', use_column_width=True)

        image_preprocessed = preprocess(image).unsqueeze(0).to(device)

        start_time = time.time()
        with torch.no_grad():
            prediction = model(image_preprocessed)
        end_time = time.time()
        prediction_time = end_time - start_time

        predicted_class = torch.argmax(prediction).item()
        confidence = torch.nn.functional.softmax(prediction, dim=1)[0][predicted_class].item()

        st.write(f"Класс: {predicted_class}")
        st.write(f"Вероятность: {confidence:.2f}")
        st.write(f"Предсказание выполнено за {prediction_time:.4f} секунд")
    except Exception as e:
        st.error(f"Ошибка при обработке изображения: {str(e)}")