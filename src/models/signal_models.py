""" Definitions of 1D Convolutional models that take 1D signals as input
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation
from tensorflow.keras.layers import Conv1D, MaxPooling1D, ZeroPadding1D, GlobalAveragePooling1D

def sig_CNN_1(input_dim=(900, 1)):
    """
    Defines a one-dimensional convolutional model
    """
    model = Sequential()
    
    model.add(Conv1D(32, 10, activation='relu', input_shape=input_dim))
    model.add(Conv1D(32, 10, activation='relu'))
    model.add(MaxPooling1D(3))
    
    model.add(Conv1D(64, 10, activation='relu'))
    model.add(Conv1D(64, 10, activation='relu'))
    model.add(GlobalAveragePooling1D())
    
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    
    return model