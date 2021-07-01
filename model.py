"""Model.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np

import keras.backend as K

# model imports
from keras.models import Sequential, Model, Input
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.layers import Lambda, BatchNormalization, ZeroPadding2D, Dropout
from tensorflow.keras.regularizers import l2


def euclidean_distance(vects):
    """Compute Euclidean Distance between two vectors.

    Euclidean distance is defined as the length of a line
    segment between the two points.

    d(p,q) = √ [Σ(qi – pi)^2]

    Args:
        vects : vectors

    Returns:
        euclidean distance between vects.
    """
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))


def eucl_dist_output_shape(shapes):
    """Returns shape.

    Args:
        shapes : shape of euclidean distance.

    Returns:
        Shape of euclidean distance.
    """
    shape1, shape2 = shapes
    return (shape1[0], 1)


def build_model(shape=(224,224,1)):
    """Build CNN siamese model.

    Args:
        shape -- tuple : input shape of (224, 224, 1).

    Returns:
        model -- keras.models.Model : siamese model.
    """

    in_left = Input(shape=shape, name="left_input")
    in_right = Input(shape=shape, name="right_input")

    # sequential network
    network = Sequential(name="sequential_network")

    # 1 convolutional
    network.add(Conv2D(96, (11, 11), activation='relu',
                       input_shape=shape,
                       strides=4,
                       kernel_initializer='glorot_uniform',
                       kernel_regularizer=l2(2e-4)))
    network.add(BatchNormalization(axis=1, momentum=0.9, epsilon=1e-06))
    network.add(MaxPooling2D((3, 3), strides=(2, 2)))
    network.add(ZeroPadding2D((2, 2)))

    # 2 convolutional
    network.add(Conv2D(256, (5, 5), activation='relu',
                       strides=1,
                       kernel_initializer='glorot_uniform',
                       kernel_regularizer=l2(2e-4)))
    network.add(BatchNormalization(axis=1, momentum=0.9, epsilon=1e-06))
    network.add(Dropout(0.3))
    network.add(MaxPooling2D((3, 3), strides=(2, 2)))
    network.add(ZeroPadding2D((2, 2)))

    # 3 convolutional
    network.add(Conv2D(384, (3, 3), activation='relu',
                       strides=1,
                       kernel_initializer='glorot_uniform',
                       kernel_regularizer=l2(2e-4)))
    network.add(ZeroPadding2D((2, 2)))

    # 4 convolutional
    network.add(Conv2D(256, (3, 3), activation='relu',
                       strides=1,
                       kernel_initializer='glorot_uniform',
                       kernel_regularizer=l2(2e-4)))
    network.add(MaxPooling2D((3, 3), strides=(2, 2)))
    network.add(Dropout(0.3))

    # flatten the output
    network.add(Flatten())
    # 1 dense layer
    network.add(Dense(1024, activation='relu',
                      kernel_regularizer=l2(1e-3),
                      kernel_initializer='glorot_uniform'))

    # 2 dense layer
    network.add(Dense(128, activation='sigmoid',
                      kernel_regularizer=l2(1e-3),
                      kernel_initializer='glorot_uniform'))

    # encodings for the 2 image inputs
    em_left = network(in_left)
    em_right = network(in_right)

    # customized layer to compute the absolute difference between the encodings
    distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)(
        [em_left, em_right])

    # connecting the inputs with the outputs
    model = Model(
        inputs=[in_left, in_right],
        outputs=distance
    )

    return model
