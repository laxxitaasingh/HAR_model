import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the TensorFlow model
model_path = 'D:\\HARmodel'  # Replace 'your_model_directory' with the path to your TensorFlow model directory
model = tf.saved_model.load(model_path)

# Function to preprocess the image before feeding it to the model
def preprocess_image(image):
    image = image.resize((224, 224))  # Assuming the model requires 224x224 images
    image = np.array(image)
    image = image / 255.0  # Normalize the pixel values to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to make predictions using the loaded model
def make_prediction(image):
    processed_image = preprocess_image(image)
    prediction = model(processed_image)  # Assuming your model is callable
    return prediction.numpy()

# Streamlit App
def main():
    st.title('Image Classification App')
    st.write('Upload an image and let the model classify it!')

    uploaded_image = st.file_uploader('Choose an image...', type=['jpg', 'jpeg', 'png'])

    if uploaded_image is not None:
        # Display the uploaded image
        image = Image.open(uploaded_image)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Make a prediction when the user clicks the 'Predict' button
        if st.button('Predict'):
            prediction = make_prediction(image)
            st.write('Prediction:', prediction)  # Display the prediction

if __name__ == '__main__':
    main()
