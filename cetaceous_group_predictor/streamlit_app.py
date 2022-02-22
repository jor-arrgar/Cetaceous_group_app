import pandas as pd
import numpy as np
import pickle
import streamlit as st
from tensorboard import summary
from tensorflow.keras.models import load_model
import cv2
from PIL import Image

# Model load
model = load_model('cetaceous_group_predictor/model.h5')
with open('cetaceous_group_predictor/pickle_trained_model_info' , 'rb') as model_file:
    history_df = pickle.load(model_file)
# Parameters tables generation
best_models = pd.read_csv('cetaceous_group_predictor/best_models.csv' , sep= ';')
model_cte = pd.read_csv('cetaceous_group_predictor/model_ctes.csv' , sep= ';')
train_cte = pd.read_csv('cetaceous_group_predictor/train_ctes.csv' , sep= ';')

st.sidebar.title('CETACEOUS IDENTIFICATION')
st.sidebar.image('https://th.bing.com/th/id/OIP.FN_C2qla_H2CoXWPMPjWeAHaE8?pid=ImgDet&rs=1')
page = st.sidebar.selectbox('Menu' , ('Presentation' , 'Predictor model info' , 'Predictor'))

if page == 'Presentation':
    st.title('CETACEOUS IMAGE PREDICTOR')
    st.image('https://coastbeat.com.au/wp-content/uploads/2018/06/whales3.jpg')
    st.write('This project is based on a image predictor, which has been trained to diferenciate different groups of cetaceous (Cetacea) by photographs taken in the moment they take the air over the sea surface.\n')
    st.write('The predictor distinguishs 3 types of cetaceous by common name: whales, dolphins and belugas. Killer whales and false killer whales are included in the dolphins group.\n')
    st.image('cetaceous_group_predictor/images_to_predict/00405189464f9d.jpg' , use_column_width= 'always' , caption= 'Kind of photograph to pass to the model. Whale (predicted).')
    st.write('The images and the project idea have been taken from a Kaggle competition: Happywhale - Whale and Dolphin Identification. This competition\'s aim is to identify specific individuals by their scars, level not reached in this project.\n')
    st.write('Cetaceous identification. By Jorge Arranz as part of Data Science Bootcamp in The Bridge Academy. 18/02/2022')

if page == 'Predictor model info':
    history = st.sidebar.checkbox('Show historic epochs values table (loss and accuracy)')

    st.header('Information and parameter used in the model')
    st.write('The image dataset is composed of 4500 photographs, 1500 per category.')
    st.subheader('Constant parameters during experimentation')
    st.write('Model preparation:')
    st.write(model_cte)
    st.write('Training preparation:')
    st.write(train_cte)
    st.subheader('Modified parameters during experiments and best models')
    st.write('Due to computational and time limitations, 108 models with 32x32 photographs and 42 models with 100x100 have been generated . In the next table, the best parameters for each group are shown:')
    st.write(best_models)
    st.header('The model selected is nº 2 (100 x 100 images)')
    st.write('This graphic shows the stats during model training:')
    st.image('cetaceous_group_predictor/training_stats.png')
    if history:
        st.write('This table contains the values of loss and accuracy for train and validation for each epoch run by the selected model.')
        st.write(history_df)

if page == 'Predictor':
    st.header('Prediction model')
    st.subheader('Uploader')
    img_file = st.file_uploader('Please, upload your images in .jpg format inside a file' , accept_multiple_files= True , type= 'jpg')
    show_table = st.sidebar.checkbox('Show complete table')
    if img_file: # Activate when upload
        img_list = list(img_file)
        name_list = []
        for img in img_list:
            name_list.append(img.name)
        select_image = st.selectbox('Image' , (name_list))
        result_list = []
        for pos , img in enumerate(name_list):
            image = img_list[pos]
            image = (Image.open(image)) 

            if select_image == img:    
                st.image(image)
            # IMAGE TRANSFORMATION
            
            img_np = np.array(image)
            mod_img = cv2.resize(img_np , (100 , 100))
            stand_img = mod_img / 255.0
            stand_img = stand_img.reshape(1 , 100 , 100 , 3)
            # PREDICTION
            predictions = model.predict(stand_img)
            prediction = np.array(predictions).reshape(1,3).round(3)
            prediction_proba_df = pd.DataFrame(prediction , index= [img] , columns= ['Dolphin' , 'Whale' , 'Beluga'])
            
            if select_image == img and not show_table:
                st.write(prediction_proba_df)
            result_list.append(prediction_proba_df)
        if show_table:
            result_df = pd.concat(result_list)
            st.write(result_df)
            #st.download_button('Download complete results' , result_df , mime= 'text/plain')
            csv = result_df.to_csv().encode('utf-8')

            st.download_button(
                label="Download complete results (CSV)",
                data=csv,
                file_name='results.csv',
                mime='text/csv')


