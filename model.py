import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np

import keras.backend as K

# model imports
from keras.models import Sequential, Model, Input
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.layers import Lambda
from keras.initializers import RandomNormal
from tensorflow.keras.regularizers import l2


def build_model(shape):
    # hyperparameters
    initialize_weights = RandomNormal(mean=0.0, stddev=0.01)
    initialize_bias = RandomNormal(mean=0.5, stddev=0.01)

    in_left = Input(shape=shape, name="left_input")
    in_right = Input(shape=shape, name="right_input")

    # Sequential network
    network = Sequential(name="sequential_network")

    # first layer - Convolutional
    network.add(Conv2D(64, (10, 10), activation='relu',
                       input_shape=shape,
                       kernel_initializer=initialize_weights,
                       kernel_regularizer=l2(2e-4)))
    network.add(MaxPooling2D())

    # second layer - Convolutional
    network.add(Conv2D(128, (7, 7), activation='relu',
                       kernel_initializer=initialize_weights,
                       bias_initializer=initialize_bias,
                       kernel_regularizer=l2(2e-4)))
    network.add(MaxPooling2D())

    # third layer - Convolutional
    network.add(Conv2D(128, (4, 4), activation='relu',
                       kernel_initializer=initialize_weights,
                       bias_initializer=initialize_bias,
                       kernel_regularizer=l2(2e-4)))
    network.add(MaxPooling2D())

    # fourth layer - Convolutional
    network.add(Conv2D(256, (4, 4), activation='relu',
                       kernel_initializer=initialize_weights,
                       bias_initializer=initialize_bias,
                       kernel_regularizer=l2(2e-4)))

    # flatten the output
    network.add(Flatten())
    network.add(Dense(512, activation='sigmoid',
                      kernel_regularizer=l2(1e-3),
                      kernel_initializer=initialize_weights,
                      bias_initializer=initialize_bias))

    # encodings for the 2 image inputs
    em_left = network(in_left)
    em_right = network(in_right)

    # customized layer to compute the absolute difference between the encodings
    L1_layer = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
    L1_distance = L1_layer([em_left, em_right])

    # final similarity score prediction
    prediction = Dense(1, activation='sigmoid',
                       bias_initializer=initialize_bias)(L1_distance)

    # connecting the inputs with the outputs
    model = Model(
        inputs=[in_left, in_right],
        outputs=prediction
    )

    return model
