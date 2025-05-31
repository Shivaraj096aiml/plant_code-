import streamlit as st
from PIL import Image
import json
import os
import torch
from model2 import load_model, predict_image

# Load the model
model = load_model()

def main():
    st.title("AGRISCAN:SMART PLANT DISEASE DETECTOR")
    st.write("Upload an image of a plant leaf to get predictions on its health and possible diseases.")

    # File uploader for images
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "bmp"])

    prediction = None  # <-- Initialize here

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_container_width=True)

        # Predict the disease
        st.write("Analyzing...")
        prediction, plant_class, disease_class, disease_name = predict_image(image, model)

    if prediction:
        st.subheader("Prediction Results")

        st.markdown(f"**Plant Name:** {prediction['Plant']['Name']}")
        st.markdown(f"**Plant Confidence:** {prediction['Plant']['Confidence']}%")
        st.markdown("**Plant Features:**")
        st.write(prediction['Plant']['Features'])

        st.markdown(f"**Disease Detected:** {prediction['Analysis']['Disease']}")
        st.markdown(f"**Disease Confidence:** {prediction['Analysis']['Confidence']}%")
        st.markdown(f"**Severity Level:** {prediction['Analysis']['Severity']['Level']}")
        st.markdown(f"**Severity Percentage:** {prediction['Analysis']['Severity']['Percentage']:.2f}%")

        st.markdown("**Leaf Health:**")
        st.write(prediction['Analysis']['LeafHealth'])

        st.subheader("Treatment Recommendations")
        for rec in prediction["Recommendations"]:
            st.write(f"- {rec}")

        st.markdown(f"**Time:** {prediction['TimeStamp']}")
    elif uploaded_file is not None:
        st.error("Error in processing the image. Please try again.")

if __name__ == "__main__":
    main()