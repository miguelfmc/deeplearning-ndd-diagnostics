"""Definition of Convolutional Neural Network CNN_4
"""

def create_model(input_dim=(11, 34, 1)):
    
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