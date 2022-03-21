# Cetaceous_group_app
Whales, dolphins and belugas identification through .jpg of the animal taking air on the sea surface using a Streamlit app.

Photographs have been taking from Kaggle (Happywhale), but only 4500 of 46000 have been used to train the model. The experiment has limitations related to time and computational capabilities, those are the reasons why only 200 different convolutional neural net have been proved. The model is displayed on a Streamlit interface with the possibility of upload your images (.jpg) and download a .csv with the results, showing the probability of each gruop per image.

If you want to repeat the experiment, you will need to downolad the image set from Kaggle (https://www.kaggle.com/c/happy-whale-and-dolphin) and extract the images referenced on model_preparation/data.csv, as they consume a large amount of memory to be uploaded to github.

Author: Jorge Arranz, as part of the Data Science Bootcamp in The Bridge Academy (Madrid). Date: 18/02/2022

App url: https://share.streamlit.io/jor-arrgar/whales_ml_0/main/cetaceous_group_predictor/streamlit_app.py
