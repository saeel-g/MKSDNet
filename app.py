import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

with st.sidebar:
    st.markdown('''
    # About
    This Model is developed as B.Tech Project
    \n Made by Saeel Gote [(GitHub)](https://github.com/saeel-g)
''')

model = load_model('Densenet2_BCE_bestmodel_2.hdf5')

def preprocess_image(image):
    image = image.convert('RGB')
    image = image.resize((256, 256))
    image = np.array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

st.title('A Multi-Kernel Sparse Dense Network (MKSDnet) for Retinal Disease risk classification')

file = st.file_uploader('Upload an image', type=['.jpg', '.png', '.jpeg'],help="Minimum Image resolution Should be 256x256 px")

if file is not None:
    # Display the uploaded image
    image = Image.open(file)
    st.image(image, caption='Uploaded Image.', use_column_width=False)

    # Preprocess the image
    processed_image = preprocess_image(image)

    # st.image(processed_image, caption='processed Image.', use_column_width=False)
    # Make predictions
    predictions = model.predict(processed_image)

    # Display the predicted class
    if np.argmax(predictions)==1:
        # st.write(predictions)
        st.write(f"No Risk")
    elif np.argmax(predictions)==0:
        # st.write(predictions)
        st.write(f"Risk")
