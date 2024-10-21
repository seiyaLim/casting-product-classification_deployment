import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms
from PIL import Image
import streamlit as st

# Load the model
@st.cache(allow_output_mutation=True)
def load_model():
    model = models.mobilenet_v2(pretrained=False)
    model.features[0][0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    
    # Load saved weights
    model.load_state_dict(torch.load('trained_model3.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

# Image preprocessing function
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    image = transform(image).unsqueeze(0)
    return image

# Prediction function
def predict(image, model):
    image = preprocess_image(image)
    output = model(image)
    _, predicted = torch.max(output, 1)
    return 'OK' if predicted.item() == 1 else 'DEFECTIVE'

# Streamlit app interface
def run():
    st.title('Casting Product Quality Inspection')

    # Image uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Load the model
        model = load_model()

        # Make a prediction
        prediction = predict(image, model)
        st.write(f"Prediction: **{prediction}**")

if __name__ == '__main__':
    run()
