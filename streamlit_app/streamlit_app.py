import logging
import numpy as np
import streamlit as st
from PIL import Image
import torchvision.transforms.functional as F
from streamlit_drawable_canvas import st_canvas
import torchvision.transforms as transforms
from utils.predict import predict
from utils.preprocess import transform_image
from utils.utils import get_model
import matplotlib.pyplot as plt

st.title('Handwritten digit detector')
logging.info('Starting')

col1, col2 = st.columns(2)

with col1:
    # Create a canvas component
    canvas_result = st_canvas(
        fill_color='#fff',
        stroke_width=30,
        stroke_color='#000',
        background_color='#fff',
        update_streamlit=True,
        height=400,
        width=400,
        drawing_mode='freedraw',
        key='canvas',
    )

with col2:
    logging.info('canvas ready')
    if canvas_result.image_data is not None:
        img = Image.fromarray(canvas_result.image_data.astype('uint8'))
        image = transform_image(img)
        # Normalize the image
        st.image(image.squeeze(0).numpy(), width=200)
        transform = transforms.Compose([
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        image = transform(image)

        normalized_image = F.to_pil_image(image.squeeze(0))
        st.image(normalized_image, width=200)
        logging.info('image augmented')
        

        model = get_model()
        logging.info('model ready')
        if st.button('Predict'):
            prediction = predict(model, image)
            st.write(f'Prediction: {prediction}')
        

text = """
This is a small app for handwritten digit recognition and recognition developed for fun. It uses a small DL model trained from scratch.
You can draw a digit (or whatever you want) and the model will try to understand what is it.
If you want to know how the app works in more detail, you are welcome to read "About" page.
Enjoy! :)
"""

st.markdown(text, unsafe_allow_html=True)