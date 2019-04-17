""" Definitions of 2D Convolutional models that take spectrograms as input

spect_CNN_1 and spec_CNN_2 are some of the first models that were used in the project
and do not yield great performance

spect_CNN_3 and spect_CNN_4 work better

"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D

def spect_CNN_1(input_dim=(11, 34, 1), n_filters_1=8, n_filters_2=16):
    """
    The function defines the CNN_1 model with the given hyperparameters
    
    Arguments
        n_filters_1:(int) hyperparameter controlling the number of filters in
            the first conv layer
        n_filters_2: (int) hyperparameter controlling the number of filters in
            the first conv layer
   
    Returns
        model: keras model
    """
    model = Sequential()
    
    model.add(ZeroPadding2D(padding=(0, 1), input_shape=input_dim))
    model.add(Conv2D(n_filters_1, (1, 3), strides=(1, 1),
                     activation='relu', use_bias=True,
                    kernel_initializer='glorot_normal', bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(1, 3))) # last column being dropped
    
    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(Conv2D(n_filters_2, (3, 3), strides=(2, 2),
                    activation='relu', use_bias=True,
                    kernel_initializer='glorot_normal', bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    
    model.add(Flatten())
    
    last_layer = model.layers[-1]
    
    model.add(Dense(int(last_layer.output_shape[-1] * 1.5), activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    return model

def spect_CNN_2(input_dim=(11, 34, 1), n_filters_1=8, n_filters_2=16):
    """
    The function defines the CNN_2 model
    
    Arguments
        n_filters_1: (int) hyperparameter controlling the number of filters in
            the first conv layer
        n_filters_2: (int) hyperparameter controlling the number of filters in
            the first conv layer
    
    Returns
        model: keras model
    """
    model = Sequential()
    
    model.add(ZeroPadding2D(padding=(0, 1), input_shape=input_dim))
    model.add(Conv2D(n_filters_1, (2, 3), strides=(1, 1),
                     activation='relu', use_bias=True,
                    kernel_initializer='glorot_normal', bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(n_filters_2, (3, 3), strides=(2, 2),
                    activation='relu', use_bias=True,
                    kernel_initializer='glorot_normal', bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Flatten())
    
    last_layer = model.layers[-1]
    
    model.add(Dense(int(last_layer.output_shape[-1] * 1.5), activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    return model

def spect_CNN_3(input_dim=(11, 34, 1), n_filters_1=8, n_filters_2=16, n_filters_3=2):
    """
    The function defines the CNN_3 model with the given hyperparameters
        
    Arguments
        n_filters_1: (int) hyperparameter controlling the number of filters in
            the first conv layer
        n_filters_2: (int) hyperparameter controlling the number of filters in
            the first conv layer
        n_filters_3: (int) hyperparameter controlling the number of filters in
            the third conv layer
        
    Returns
        model: keras model
    """
    model = Sequential()
    
    model.add(Conv2D(n_filters_1, (3, 3), strides=(1, 1), padding='same',
                     activation='relu', use_bias=True,
                    kernel_initializer='glorot_normal', bias_initializer='zeros',
                    input_shape=input_dim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(n_filters_2, (3, 3), strides=(1, 1), padding='same',
                    activation='relu', use_bias=True,
                    kernel_initializer='glorot_normal', bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # 1x1 convolutions to reduce dimensionality
    model.add(Conv2D(n_filters_3, (1, 1), strides=(1, 1),
                    activation='relu', use_bias=True,
                    kernel_initializer='glorot_normal', bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Flatten())
    
    last_layer = model.layers[-1]
    
    model.add(Dense(int(last_layer.output_shape[-1] * 1.5), activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    return model

def spect_CNN_4(input_dim=(11, 34, 1)):
    """
    Defines the CNN_4 model
    """
    tu_x, tu_y, _ = input_dim
    
    model = Sequential()
    model.add(Conv2D(32, (5, 5), padding='same', input_shape=(tu_x, tu_y, 1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (5, 5), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (5, 5), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    
    return model