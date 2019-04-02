"""Definition of Convolutional Neural Network CNN_3
"""

def create_model(input_dim=(11, 34, 1), n_filters_1=8, n_filters_2=16, n_filters_3=2):
    """
    The function defines a Convolutional Neural Network model with the given hyperparameters
        
    Arguments:
    n_filters_1 -- (int) hyperparameter controlling the number of filters in
        the first conv layer
    n_filters_2 -- (int) hyperparameter controlling the number of filters in
        the first conv layer
    n_filters_3 -- (int) hyperparameter controlling the number of filters in
        the third conv layer
        
    Returns:
    model -- keras model
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