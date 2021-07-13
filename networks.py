"""The different networks implemented.
"""
from utils import *

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np

import tensorflow as tf
import keras.backend as K

# model imports
from keras.models import Sequential, Model
from keras.layers import Input, concatenate, Layer
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.layers import Lambda, BatchNormalization, ZeroPadding2D, Dropout
from tensorflow.keras.regularizers import l2
from keras.optimizers import Adam, RMSprop
from keras.metrics import Mean

rms = RMSprop(lr=1e-4, rho=0.9, epsilon=1e-08)


def embedding_net(shape=(224, 224, 1)):
    """Embeddnig network.

    Args:
        shape -- tuple : input shape of (224, 224, 1).

    Returns:
        network -- keras.models.Sequential : embedding sequential network.
    """
    # Sequential network
    network = Sequential(name="sequential_network")

    # 1 Conv2D
    network.add(Conv2D(96, (11, 11), activation='relu',
                       input_shape=shape,
                       strides=4,
                       kernel_initializer='glorot_uniform',
                       kernel_regularizer=l2(2e-4)))
    network.add(BatchNormalization(axis=1, momentum=0.9, epsilon=1e-06))
    network.add(MaxPooling2D((3, 3), strides=(2, 2)))
    network.add(ZeroPadding2D((2, 2)))

    # 2 Conv2D
    network.add(Conv2D(256, (5, 5), activation='relu',
                       strides=1,
                       kernel_initializer='glorot_uniform',
                       kernel_regularizer=l2(2e-4)))
    network.add(BatchNormalization(axis=1, momentum=0.9, epsilon=1e-06))
    network.add(Dropout(0.3))
    network.add(MaxPooling2D((3, 3), strides=(2, 2)))
    network.add(ZeroPadding2D((2, 2)))

    # 3 Conv2D
    network.add(Conv2D(384, (3, 3), activation='relu',
                       strides=1,
                       kernel_initializer='glorot_uniform',
                       kernel_regularizer=l2(2e-4)))
    network.add(ZeroPadding2D((2, 2)))

    # 4 Conv2D
    network.add(Conv2D(256, (3, 3), activation='relu',
                       strides=1,
                       kernel_initializer='glorot_uniform',
                       kernel_regularizer=l2(2e-4)))
    network.add(MaxPooling2D((3, 3), strides=(2, 2)))
    network.add(Dropout(0.3))

    # flatten the output to 1D
    network.add(Flatten())

    return network


# ###################################################################################################
def pairs_net(shape=(224, 224, 1)):
    """Build custom siamese CNN model with contrastive loss.

    Args:
        shape -- tuple : input shape of (224, 224, 1).

    Returns:
        model -- keras.models.Model : siamese CNN model.
    """
    # pair of inputs
    in_left = Input(shape=shape, name="left_input")
    in_right = Input(shape=shape, name="right_input")

    network = embedding_net(shape)

    # 1 Dense
    network.add(Dense(1024, activation='relu',
                      kernel_regularizer=l2(1e-3),
                      kernel_initializer='glorot_uniform'))

    # sigmoid activation
    network.add(Dense(128, activation='sigmoid',
                      kernel_regularizer=l2(1e-3),
                      kernel_initializer='glorot_uniform'))

    # encodings for the 2 image inputs
    em_left = network(in_left)
    em_right = network(in_right)

    # final customized Lambda layer
    # to compute the absolute difference between the encodings
    distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)(
        [em_left, em_right])

    # connecting the inputs with the outputs
    model = Model(
        inputs=[in_left, in_right],
        outputs=distance
    )

    model.compile(optimizer=rms, loss=contrastive_loss)

    return model


# ###################################################################################################
class TripletDistanceLayer(Layer):
    """Layer for computing distances.
    
    Layer for calculating distances between the anchor
    embedding and the positive embedding, and the anchor
    embedding and the negative embedding.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, anchor, positive, negative):
        ap_distance = tf.reduce_sum(tf.square(anchor - positive), -1)
        an_distance = tf.reduce_sum(tf.square(anchor - negative), -1)

        return (ap_distance, an_distance)


def triplet_net(shape=(224, 224, 1)):
    """Build custom triplet loss CNN model.

    Args:
        shape -- tuple : input shape of (224, 224, 1).

    Returns:
        model -- keras.models.Model : triplet loss CNN model.
    """
    # triplet inputs
    in_anchor = Input(shape=shape, name="anchor_input")
    in_positive = Input(shape=shape, name="positive_input")
    in_negative = Input(shape=shape, name="negative_input")

    network = embedding_net(shape)

    # 1 Dense
    network.add(Dense(512, activation="relu",
                      kernel_regularizer=l2(1e-3),
                      kernel_initializer='glorot_uniform'))
    network.add(BatchNormalization())

    # 2 Dense
    network.add(Dense(256, activation="relu",
                      kernel_regularizer=l2(1e-3),
                      kernel_initializer='glorot_uniform'))
    network.add(BatchNormalization())

    # 3 Dense
    network.add(Dense(256))

    # encodings for the 3 image inputs and calculating distances
    distances = TripletDistanceLayer()(
        network(in_anchor),
        network(in_positive),
        network(in_negative),
    )

    # connecting the inputs with the outputs
    model = Model(
        inputs=[in_anchor, in_positive, in_negative],
        outputs=distances
    )

    return model


class TripletModel(Model):
    """The Siamese Network model with a custom training and testing loops.

    Computes the triplet loss using the three embeddings produced by the
    Siamese Network.

    The triplet loss is defined as:
       L(A, P, N) = max(‖f(A) - f(P)‖² - ‖f(A) - f(N)‖² + margin, 0)
    """

    def __init__(self, siamese_network, margin=0.5):
        super(TripletModel, self).__init__()
        self.siamese_network = siamese_network
        self.margin = margin
        self.loss_tracker = Mean(name="loss")

    def call(self, inputs):
        return self.siamese_network(inputs)

    def train_step(self, data):
        # GradientTape to compute the loss so we can get the gradients.
        with tf.GradientTape() as tape:
            loss = self._compute_loss(data)

        # Storing the gradients of the loss function with respect to the
        # weights/parameters.
        gradients = tape.gradient(loss, self.siamese_network.trainable_weights)

        # Applying the gradients on the model using the specified optimizer
        self.optimizer.apply_gradients(
            zip(gradients, self.siamese_network.trainable_weights)
        )

        # Update and return the training loss metric.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        loss = self._compute_loss(data)

        # Update and return the loss metric.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def _compute_loss(self, data):
        # The output of the network is a tuple containing the distances
        ap_distance, an_distance = self.siamese_network(data)

        # Computing the Quadruplet Loss
        loss = ap_distance - an_distance
        loss = tf.maximum(loss + self.margin, 0.0)
        return loss

    @property
    def metrics(self):
        return [self.loss_tracker]


# ###################################################################################################
class QuadrupletDistanceLayer(Layer):
    """Layer for computing distances.
    
    Layer for calculating distances between the anchor
    embedding and the positive embedding, the anchor
    embedding and the negative embedding, and the two
    negative embeddings.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, anchor, positive, negative, negative2):
        ap_distance = tf.reduce_sum(tf.square(anchor - positive), -1)
        an_distance = tf.reduce_sum(tf.square(anchor - negative), -1)
        nn2_distance = tf.reduce_sum(tf.square(negative - negative2), -1)

        return (ap_distance, an_distance, nn2_distance)


def quadruplet_net(shape=(224, 224, 1)):
    """Build custom quadruplet loss CNN model.

    Args:
        shape -- tuple : input shape of (224, 224, 1).

    Returns:
        model -- keras.models.Model : quadruplet loss CNN model.
    """
    # quadruplet inputs
    in_anchor = Input(shape=shape, name="anchor_input")
    in_positive = Input(shape=shape, name="positive_input")
    in_negative = Input(shape=shape, name="negative_input")
    in_negative2 = Input(shape=shape, name="negative2_input")

    network = embedding_net(shape)

    # encodings for the 4 image inputs and calculating distances
    distances = QuadrupletDistanceLayer()(
        network(in_anchor),
        network(in_positive),
        network(in_negative),
        network(in_negative2),
    )

    # connecting the inputs with the outputs
    siamese_network = Model(
        inputs=[in_anchor, in_positive, in_negative, in_negative2],
        outputs=distances
    )

    return siamese_network


class QuadrupletModel(Model):
    """The Siamese Network model with a custom training and testing loops.

    Computes the quadruplet loss using the four embeddings produced by the
    Siamese Network.

    The quadruplet loss is defined as:
       L(A, P, N, N2) = max(‖f(A) - f(P)‖² - ‖f(A) - f(N)‖² + margin1, 0) +
                        max(‖f(A) - f(P)‖² - ‖f(N) - f(N2)‖² + margin2, 0)
    """

    def __init__(self, siamese_network, margin1=1, margin2=0.5):
        super(QuadrupletModel, self).__init__()
        self.siamese_network = siamese_network
        self.margin1 = margin1
        self.margin2 = margin2
        self.loss_tracker = Mean(name="loss")

    def call(self, inputs):
        return self.siamese_network(inputs)

    def train_step(self, data):
        # GradientTape to compute the loss so we can get the gradients.
        with tf.GradientTape() as tape:
            loss = self._compute_loss(data)

        # Storing the gradients of the loss function with respect to the
        # weights/parameters.
        gradients = tape.gradient(loss, self.siamese_network.trainable_weights)

        # Applying the gradients on the model using the specified optimizer
        self.optimizer.apply_gradients(
            zip(gradients, self.siamese_network.trainable_weights)
        )

        # Update and return the training loss metric.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        loss = self._compute_loss(data)

        # Update and return the loss metric.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def _compute_loss(self, data):
        # The output of the network is a tuple containing the distances
        ap_distance, an_distance, nn2_distance = self.siamese_network(data)

        # Computing the Quadruplet Loss
        term1 = tf.maximum(ap_distance - an_distance + self.margin1, 0.0)
        term2 = tf.maximum(ap_distance - nn2_distance + self.margin2, 0.0)
        loss = term1 + term2
        return loss

    @property
    def metrics(self):
        return [self.loss_tracker]
