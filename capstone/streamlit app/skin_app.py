import matplotlib.pyplot as plt
import numpy as np
import os 
import tensorflow as tf
import streamlit as st
import pandas as pd
from PIL import Image, ImageOps
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from keras.applications import ResNet50
from keras.applications.densenet import preprocess_input
import tempfile 


#to ignore file uploading warning 
st.set_option('deprecation.showfileUploaderEncoding', False)

#load model while using cache as to not reload model everytime it rus
@st.cache(allow_output_mutation = True)
def load_model(model):
    return tf.keras.models.load_model(model)

#put model path here
model = load_model('06_model')

#Title and description on page 
st.write('''# Have a Skin Concern? \n #### This application can help you decide if your skin concern resembles other diagnosed skin conditions. \n **This application should not be used in place of consulting your doctor. This project is for educational and purely academic purposes only. Please do not use this project as your own diagnostic tool. Consult your doctor regardless of the result. The result IS NOT a medical diagnosis**
''')

@st.cache
def load_image(image_file):
    img = Image.open(image_file)
    return img

#making a area to upload image to put through model 
file=st.file_uploader('Please upload an image of your skin concern', type=['jpeg', 'png', 'jpg'])

if file is None:
    st.stop()

if file is not None:
    file_details = {"FileName":f'{os.path.getctime(os.getcwd())}{file.name}',"FileType":file.type}
    # st.write(file_details)
    img = load_image(file)
    # st.image(img)
    with open(os.path.join('tempDir/benign',file_details['FileName']),"wb") as f: 
        f.write(file.getbuffer())         
    st.success('saved_file')
    
#function to import and predict 
def import_and_predict(image_data, model):
    test_batches=ImageDataGenerator(rescale=1./255, preprocessing_function=tf.keras.applications.resnet50.preprocess_input).flow_from_directory(directory='tempDir', 
    target_size=(512,512), 
    classes=['benign', 'malignant'],
    batch_size=2, 
    shuffle=False)
    prediction=model.predict(test_batches)
    
    return prediction 

if file is None:
    st.text('Please upload an image file')
    #st.stop()
else:
    image=Image.open(file)
    st.image(image, use_column_width=True)
    predictions=import_and_predict(image, model)
    class_names=['NOT OF CONCERN. Consult your doctor anyways', 'OF CONCERN. Please consult your doctor']
    beginning_string='This image is: '+class_names[np.argmax(predictions[-2])]
    st.success(beginning_string)
