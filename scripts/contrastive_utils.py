"""Contrastive loss utility functions.

embedding_net
build_contrastive_model

compute_accuracy_roc
evaluation_plots
draw_eval_contrastive
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import math
import numpy as np

import tensorflow as tf
import keras.backend as K

# model imports
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.layers import Lambda, BatchNormalization, ZeroPadding2D, Dropout
from tensorflow.keras.regularizers import l2

from sklearn.metrics import roc_curve, roc_auc_score

import matplotlib.pyplot as plt


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


def contrastive_loss(y_true, y_pred):
    """Contrastive loss.

    if y = true and d = pred,
    d(y,d) = mean(y * d^2 + (1-y) * (max(margin-d, 0))^2)

    Args:
        y_true : true values.
        y_pred : predicted values.

    Returns:
        contrastive loss
    """
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


# MODELS ########################################################################################
def embedding_net(shape=(224, 224, 1)):
    """Embedding network.

    Args:
        shape -- tuple : input shape of (224, 224, 1).

    Returns:
        network -- keras.models.Sequential : embedding sequential network.
    """
    # Sequential embedding network
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

    # 1 Dense
    network.add(Dense(1024, activation='relu',
                      kernel_regularizer=l2(1e-3),
                      kernel_initializer='glorot_uniform'))

    # sigmoid activation
    network.add(Dense(128, activation='sigmoid',
                      kernel_regularizer=l2(1e-3),
                      kernel_initializer='glorot_uniform'))

    return network


def build_contrastive_model(shape=(224, 224, 1)):
    """Build custom siamese CNN model with contrastive loss.

    Args:
        shape -- tuple : input shape of (224, 224, 1).

    Returns:
        model -- keras.models.Model : siamese CNN model.
    """
    # pair of inputs
    left_input = Input(shape=shape, name="left_input")
    right_input = Input(shape=shape, name="right_input")

    network = embedding_net(shape)

    # encodings for the 2 image inputs
    encoded_l = network(left_input)
    encoded_r = network(right_input)

    # final customized Lambda layer
    # to compute the absolute difference between the encodings
    distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)(
        [encoded_l, encoded_r])

    # connecting the inputs with the outputs
    network_train = Model(
        inputs=[left_input, right_input],
        outputs=distance
    )

    return network_train


# EVALUATION ######################################################################################
def compute_accuracy_roc(predictions, labels):
    """Compute ROC accuracyand threshold.

    Also, plot FAR-FRR curves and P-R curves for input data.
    
    Args:
        predictions -- np.array : array of predictions.
        labels -- np.array : true labels (0 or 1).
        plot_far_frr -- bool : plots curves of True.
    
    Returns:
        max_acc -- float : maximum accuracy of model.
        best_thresh --float : best threshold for the model.
    """
    dmax = np.max(predictions)
    dmin = np.min(predictions)

    nsame = np.sum(labels == 1)  # similar
    ndiff = np.sum(labels == 0)  # different

    step = 0.001
    max_acc = 0
    best_thresh = -1

    tpr_plot = []
    frr_plot = []
    far_plot = []
    pr_plot = []
    re_plot = []

    ds = []
    for d in np.arange(dmin, dmax+step, step):
        idx1 = predictions.ravel() <= d  # guessed genuine
        idx2 = predictions.ravel() > d  # guessed forged

        tp = float(np.sum(labels[idx1] == 1))
        tn = float(np.sum(labels[idx2] == 0))
        fp = float(np.sum(labels[idx1] == 0))
        fn = float(np.sum(labels[idx2] == 1))

        tpr = float(np.sum(labels[idx1] == 1)) / nsame
        tnr = float(np.sum(labels[idx2] == 0)) / ndiff

        acc = 0.5 * (tpr + tnr)
        pr = tp / (tp + fp)
        re = tp / (tp + fn)

        if (acc > max_acc):
            max_acc, best_thresh = acc, d

        far = fp / (fp + tn)
        frr = fn / (fn + tp)

        tpr_plot.append(tpr)
        frr_plot.append(frr)
        pr_plot.append(pr)
        re_plot.append(re)
        far_plot.append(far)
        ds.append(d)

    plot_metrics = [ds, tpr_plot, far_plot, frr_plot, pr_plot, re_plot]

    return max_acc, best_thresh, plot_metrics


def find_nearest(array, value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return array[idx-1], idx-1
    else:
        return array[idx], idx


def draw_roc(fpr, tpr, threshold, auc, n_iteration):
    # find threshold
    targetfpr = 1e-3
    _, idx = find_nearest(fpr, targetfpr)
    recall = tpr[idx]

    # plot no skill
    plt.plot([0, 1], [0, 1], linestyle='--')
    # plot the roc curve for the model
    plt.plot(fpr, tpr, marker='.')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('AUC: {0:.3f} @ {4} iterations\nSensitivity : {2:.1%} @FPR={1:.0e}\nThreshold={3})'.format(
        auc, targetfpr, recall, abs(threshold), n_iteration
    ))
    # show the plot
    plt.show()


def draw_eval_contrastive(network, pairs, targets):
    pairs = [pairs[0], pairs[1]]
    pred = network.predict(pairs)
    acc, thresh, plot_metrics = compute_accuracy_roc(pred, targets)
    tpr, fpr = plot_metrics[1], plot_metrics[2]
    draw_roc(fpr, tpr, thresh, acc, 10000)
