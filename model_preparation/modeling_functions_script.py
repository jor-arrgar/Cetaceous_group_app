import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation


# Image set read and shape settings
def read_images(dataset , pixels):
    image_list = []
    for image in dataset['image']:
        path = './cetaceous_images/' + image
        img = cv2.imread(path)
        image = cv2.resize(img, (pixels, pixels))
        image_list.append(image)
    return image_list

# Train, validation and test data
def train_test(dataset , image_list):

    X = ((np.array(image_list)) / 255)
    y = to_categorical(dataset['categories'])
    
    X_train , X_test , y_train , y_test = train_test_split(X , y , test_size= 0.2 , random_state= 98)

    X_val = X_train[-600:]
    y_val = y_train[-600:]
    X_train = X_train[:-600]
    y_train = y_train[:-600]

    return X_train , X_val , X_test , y_train , y_val , y_test

def model_params(pixel , pool_size_1 , pool_size_2 , pool_size_3, n_neurons):
    model = Sequential()

    model.add(Conv2D(32, (3,3), activation='relu', 
                input_shape=(pixel, pixel, 3)))
    model.add(MaxPooling2D(pool_size=pool_size_1))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=pool_size_2))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=pool_size_3))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(n_neurons, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3,activation='softmax')) 



    model.compile(optimizer="adam",
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    
    print('-----------Modelo ==> ' , pixel , pool_size_1 , pool_size_2 , pool_size_3 , n_neurons , '---------------')
    
    return model
        
# Model training

def model_training(model , epoch , batch , X_train , y_train , X_val , y_val):

    history = model.fit(X_train,
          y_train,
          epochs=epoch,
          batch_size=batch, 
          validation_data = (X_val,y_val))
    print('---------------------------Training params ==> ' , epoch , batch , '---------------------------------------')

    return history , model
