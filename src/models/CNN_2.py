"""Definition of the Conv. Neural Network CNN_2
"""

def create_model(input_dim=(11, 34, 1), n_filters_1=8, n_filters_2=16):
    """
    The function defines the CNN_2 model
    
    Arguments:
    n_filters_1 -- (int) hyperparameter controlling the number of filters in
        the first conv layer
    n_filters_2 -- (int) hyperparameter controlling the number of filters in
        the first conv layer
    
    Returns:
    model -- keras model
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