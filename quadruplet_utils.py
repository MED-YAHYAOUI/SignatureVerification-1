"""Quadruplet loss utility functions.

embedding_net
build_metric_network
QuadrupletLossLayer
build_quadruplet_model

compute_l2_dist
compute_probs
compute_metrics
find_nearest
draw_roc
draw_eval_quadruplets
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import math
import numpy as np
from tqdm import tqdm

import keras.backend as K

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, concatenate, Layer
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Lambda, Flatten, Dense, Concatenate
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.regularizers import l2

from sklearn.metrics import roc_curve, roc_auc_score

import matplotlib.pyplot as plt


# Model ##################################################################################
def embedding_net(embeddingsize, input_shape=(224, 224, 1)):
    """Embedding network.

    Args:
        embeddingsize -- int : embedding size.
        input_shape -- tuple : input shape of (224, 224, 1).

    Returns:
        embedding -- keras.models.Sequential : embedding sequential network.
    """

    # Convolutional Neural Network
    network = Sequential(name="sequential_network")

    # 1 Conv2D
    network.add(Conv2D(128, (7, 7), activation='relu',
                       padding='same',
                       input_shape=input_shape,
                       kernel_initializer='he_uniform',
                       kernel_regularizer=l2(2e-4)))
    network.add(MaxPooling2D())

    # 2 Conv2D
    network.add(Conv2D(128, (5, 5), activation='relu',
                       padding='same',
                       kernel_initializer='he_uniform',
                       kernel_regularizer=l2(2e-4)))
    network.add(MaxPooling2D())

    # 3 Conv2D
    network.add(Conv2D(64, (5, 5), activation='relu',
                       padding='same',
                       kernel_initializer='he_uniform',
                       kernel_regularizer=l2(2e-4)))

    # flatten the output to 1D
    network.add(Flatten())

    # 1 Dense
    network.add(Dense(2048, activation='relu',
                      kernel_regularizer=l2(1e-3),
                      kernel_initializer='he_uniform'))

    # 2 Dense
    network.add(Dense(embeddingsize, activation=None,
                      kernel_regularizer=l2(1e-3),
                      kernel_initializer='he_uniform'))

    # Force the encoding to live on the d-dimentional hypershpere
    network.add(Lambda(lambda x: K.l2_normalize(x, axis=-1)))

    return network


def build_metric_network(single_embedding_shape):
    '''
    Define the neural network to learn the metric
    Input : 
            single_embedding_shape : shape of input embeddings or feature map. Must be an array

    '''
    # compute shape for input
    input_shape = single_embedding_shape
    # the two input embeddings will be concatenated
    input_shape[0] = input_shape[0]*2

    # Neural Network
    network = Sequential(name="learned_metric")
    network.add(Dense(10, activation='relu',
                      input_shape=input_shape,
                      kernel_regularizer=l2(1e-3),
                      kernel_initializer='he_uniform'))
    network.add(Dense(10, activation='relu',
                      kernel_regularizer=l2(1e-3),
                      kernel_initializer='he_uniform'))
    network.add(Dense(10, activation='relu',
                      kernel_regularizer=l2(1e-3),
                      kernel_initializer='he_uniform'))

    # Last layer : binary softmax
    network.add(Dense(2, activation='softmax'))

    # Select only one output value from the softmax
    network.add(Lambda(lambda x: x[:, 0]))

    return network


class QuadrupletLossLayer(Layer):
    def __init__(self, alpha=1, beta=0.5, **kwargs):
        self.alpha = alpha
        self.beta = beta
        self.debugeric = 1

        super(QuadrupletLossLayer, self).__init__(**kwargs)

    def quadruplet_loss(self, inputs):
        ap_dist, an_dist, nn_dist = inputs

        # square
        ap_dist2 = K.square(ap_dist)
        an_dist2 = K.square(an_dist)
        nn_dist2 = K.square(nn_dist)

        return K.sum(K.maximum(ap_dist2 - an_dist2 + self.alpha, 0), axis=0) + K.sum(K.maximum(ap_dist2 - nn_dist2 + self.beta, 0), axis=0)

    def call(self, inputs):
        loss = self.quadruplet_loss(inputs)
        self.add_loss(loss)
        return loss


def build_quadruplet_model(input_shape, network, metricnetwork, margin=1, margin2=0.5):
    '''
    Define the Keras Model for training 
        Input : 
            input_shape : shape of input images
            network : Neural network to train outputing embeddings
            metricnetwork : Neural network to train the learned metric
            margin : minimal distance between Anchor-Positive and Anchor-Negative for the lossfunction (alpha1)
            margin2 : minimal distance between Anchor-Positive and Negative-Negative2 for the lossfunction (alpha2)

    '''
    # Define the tensors for the four input images
    anchor_input = Input(input_shape, name="anchor_input")
    positive_input = Input(input_shape, name="positive_input")
    negative_input = Input(input_shape, name="negative_input")
    negative2_input = Input(input_shape, name="negative2_input")

    # Generate the encodings (feature vectors) for the four images
    encoded_a = network(anchor_input)
    encoded_p = network(positive_input)
    encoded_n = network(negative_input)
    encoded_n2 = network(negative2_input)

    # compute the concatenated pairs
    encoded_ap = Concatenate(
        axis=-1, name="Anchor-Positive")([encoded_a, encoded_p])
    encoded_an = Concatenate(
        axis=-1, name="Anchor-Negative")([encoded_a, encoded_n])
    encoded_nn = Concatenate(
        axis=-1, name="Negative-Negative2")([encoded_n, encoded_n2])

    # compute the distances AP, AN, NN
    ap_dist = metricnetwork(encoded_ap)
    an_dist = metricnetwork(encoded_an)
    nn_dist = metricnetwork(encoded_nn)

    # QuadrupletLoss Layer
    loss_layer = QuadrupletLossLayer(alpha=margin, beta=margin2, name='4xLoss')([
        ap_dist, an_dist, nn_dist])

    # Connect the inputs with the outputs
    network_train = Model(
        inputs=[anchor_input, positive_input, negative_input, negative2_input],
        outputs=loss_layer)

    # return the model
    return network_train


# EVALUATION ##################################################################################
def compute_l2_dist(a, b):
    return np.sum(np.square(a-b))


def compute_probs(network, X):
    '''
    Input
        network : current NN to compute embeddings.
        X : tensor of shape (m, w, h, 1) containing pics to evaluate.
        Y : tensor of shape (m,) containing true class.

    Returns
        probs : array of shape (m, m) containing distances.

    '''
    left = X[0]
    right = X[1]

    m = left.shape[0]
    probs = np.zeros((m))

    for i in tqdm(range(m), desc='QUADRUPLETS PROBS'):
        emb_left = network.predict(left[m].reshape(1, 224, 224, 1))
        emb_right = network.predict(right[m].reshape(1, 224, 224, 1))
        probs[i] = -compute_l2_dist(emb_left, emb_right)

    return probs


def compute_metrics(yprobs, probs):
    '''
    Returns
        fpr : Increasing false positive rates such that element i is the false positive rate of predictions with score >= thresholds[i]
        tpr : Increasing true positive rates such that element i is the true positive rate of predictions with score >= thresholds[i].
        thresholds : Decreasing thresholds on the decision function used to compute fpr and tpr. thresholds[0] represents no instances being predicted and is arbitrarily set to max(y_score) + 1
        auc : Area Under the ROC Curve metric
    '''
    # calculate AUC
    auc = roc_auc_score(yprobs, probs)

    # calculate roc curve
    fpr, tpr, thresholds = roc_curve(yprobs, probs)

    return fpr, tpr, thresholds, auc


def find_nearest(array, value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return array[idx-1], idx-1
    else:
        return array[idx], idx


def draw_roc(fpr, tpr, thresholds, auc, n_iteration):
    # find threshold
    targetfpr = 1e-3
    _, idx = find_nearest(fpr, targetfpr)
    threshold = thresholds[idx]
    recall = tpr[idx]

    # plot no skill
    plt.plot([0, 1], [0, 1], linestyle='--')
    # plot the roc curve for the model
    plt.plot(fpr, tpr, marker='.')
    plt.title('AUC: {0:.3f} @ {4} iterations\nSensitivity : {2:.1%} @FPR={1:.0e}\nThreshold={3})'.format(
        auc, targetfpr, recall, abs(threshold), n_iteration
    ))
    # show the plot
    plt.show()


def draw_eval_quadruplets(network, n_iteration, X, Y):
    yprobs = Y
    probs = compute_probs(network, X, Y)
    fpr, tpr, thresholds, auc = compute_metrics(yprobs, probs)
    draw_roc(fpr, tpr, thresholds, auc, n_iteration)
