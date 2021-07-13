"""Utility functions."""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import keras.backend as K


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
    """Compute contrastive loss.
    
    if y = true and d = pred,
    d(y,d) = mean(y * d^2 + (1-y) * (max(margin-d, 0))^2)
    
    Args:
        y_true : true values.
        y_pred : predicted values.

    Returns:
        contrastive loss.
    """
    margin = 1

    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0.0))

    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)


def triplet_loss(y_true, y_pred):
    """Compute triplet loss.
    
    Args:
        y_true : true values.
        y_pred : predicted values.

    Returns:
        triplet loss.
    """
    alpha = 1
    anchor, positive, negative = y_pred

    ap_dist = K.mean(K.square(anchor - positive), axis=-1)
    an_dist = K.mean(K.square(anchor - negative), axis=-1)

    return K.mean(K.maximum(0.0, ap_dist - an_dist + alpha))


def quadruplet_loss(y_true, y_pred):
    """Compute quadruplet loss.
    
    Args:
        y_true : true values.
        y_pred : predicted values.

    Returns:
        quadruplet loss.
    """
    alpha1 = 1
    alpha2 = 0.25
    anchor, positive, negative, negative2 = y_pred

    ap_dist = K.mean(K.square(anchor - positive), axis=-1)
    an_dist = K.mean(K.square(anchor - negative), axis=-1)
    nn2_dist = K.mean(K.square(negative - negative2), axis=-1)

    first_term = K.mean(
        K.maximum(0.0, ap_dist - an_dist + alpha1)
    )
    second_term = K.mean(
        K.maximum(0.0, ap_dist - nn2_dist + alpha2)
    )

    return first_term + second_term
