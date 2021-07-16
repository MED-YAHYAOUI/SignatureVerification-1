"""Triplet loss utility functions.

embedding_net
TripletLossLayer
build_triplet_model

compute_l2_dist
compute_probs
compute_metrics
find_nearest
draw_roc
draw_eval_triplets
"""
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Lambda, Flatten, Dense
from tensorflow.keras.layers import Input, concatenate, Layer
from tensorflow.keras.models import Sequential, Model
import keras.backend as K
from tqdm import tqdm
import numpy as np
import math
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# model imports


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


class TripletLossLayer(Layer):
    def __init__(self, alpha=1, **kwargs):
        self.alpha = alpha
        super(TripletLossLayer, self).__init__(**kwargs)

    def triplet_loss(self, inputs):
        anchor, positive, negative = inputs
        p_dist = K.sum(K.square(anchor-positive), axis=-1)
        n_dist = K.sum(K.square(anchor-negative), axis=-1)
        return K.sum(K.maximum(p_dist - n_dist + self.alpha, 0), axis=0)

    def call(self, inputs):
        loss = self.triplet_loss(inputs)
        self.add_loss(loss)
        return loss


def build_triplet_model(input_shape, network, margin=1):
    """Define the Keras Model for training 

    Args : 
        input_shape -- tuple : shape of input images.
        network : Neural network to train outputing embeddings.
        margin -- float : minimal distance between Anchor-Positive and
                          Anchor-Negative for the lossfunction (alpha).
    """
    # Define the tensors for the three input images
    anchor_input = Input(input_shape, name="anchor_input")
    positive_input = Input(input_shape, name="positive_input")
    negative_input = Input(input_shape, name="negative_input")

    # Generate the encodings (feature vectors) for the three images
    encoded_a = network(anchor_input)
    encoded_p = network(positive_input)
    encoded_n = network(negative_input)

    # TripletLoss Layer
    loss_layer = TripletLossLayer(alpha=margin, name='3xLoss')(
        [encoded_a, encoded_p, encoded_n])

    # Connect the inputs with the outputs
    network_train = Model(
        inputs=[anchor_input, positive_input, negative_input],
        outputs=loss_layer
    )

    # return the model
    return network_train


# EVALUATION ######################################################################################
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

    for i in tqdm(range(m), desc='TRIPLETS PROBS'):
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


def draw_eval_triplets(network, n_iteration, X, Y):
    yprobs = Y
    probs = compute_probs(network, X)
    fpr, tpr, thresholds, auc = compute_metrics(yprobs, probs)
    draw_roc(fpr, tpr, thresholds, auc, n_iteration)
