#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 12:09:17 2019

@author: admin
"""

from keras import backend as K
from keras.metrics import categorical_accuracy, categorical_crossentropy, mean_absolute_error
from sklearn.metrics import fbeta_score
from itertools import product

def accuracy_abs(y_true, y_pred):
    y_true = y_true [:,2:]
    return categorical_accuracy(y_true, y_pred)     

def accuracy_w(y_true, y_pred):
    weights = y_true[:,0]    
    y_true_val = y_true [:,2:]
    # categorical_accuracy returns a tensor with BATCH_SIZE rows and 1 column having values of 1 (correct pred) or 0 (incorrect pred)
    # we now mutliplu each elements by a weight instead of a 1
    return weights * categorical_accuracy(y_true_val, y_pred)   

def categorical_crossentropy_abs(y_true, y_pred):
    y_true = y_true [:,2:]
    return categorical_crossentropy(y_true, y_pred)     

def categorical_focal_loss(gamma=2., alpha=.25):
    """
    Softmax version of focal loss.
           m
      FL = ∑  -alpha * (1 - p_o,c)^gamma * y_o,c * log(p_o,c)
          c=1
      where m = number of classes, c = class and o = observation
    Parameters:
      alpha -- the same as weighing factor in balanced cross entropy
      gamma -- focusing parameter for modulating factor (1-p)
    Default value:
      gamma -- 2.0 as mentioned in the paper
      alpha -- 0.25 as mentioned in the paper
    References:
        Official paper: https://arxiv.org/pdf/1708.02002.pdf
        https://www.tensorflow.org/api_docs/python/tf/keras/backend/categorical_crossentropy
    Usage:
     model.compile(loss=[categorical_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    def categorical_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred: A tensor resulting from a softmax
        :return: Output tensor.
        """
        y_true = y_true [:,2:]

        # Scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)

        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

        # Calculate Cross Entropy
        cross_entropy = -y_true * K.log(y_pred)

        # Calculate Focal Loss
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy

        # Sum the losses in mini_batch
        return K.sum(loss, axis=1)

    return categorical_focal_loss_fixed

def w_categorical_focal_loss(gamma=2., alpha=.25):
    def w_categorical_focal_loss_fixed(y_true, y_pred):
        weights = y_true[:,1]    
        y_true = y_true [:,2:]

        # Scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)

        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

        # Calculate Cross Entropy
        cross_entropy = -y_true * K.log(y_pred)

        # Calculate Focal Loss
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy

        # Sum the losses in mini_batch
        return K.sum(loss, axis=1) * weights
    return w_categorical_focal_loss_fixed

# matrix one
def w_categorical_crossentropy(y_true, y_pred, weights):
    y_true = y_true [:,2:]
    nb_cl = len(weights)
    final_mask = K.zeros_like(y_pred[:, 0])
    y_pred_max = K.max(y_pred, axis=1)
    y_pred_max = K.expand_dims(y_pred_max, 1)
    y_pred_max_mat = K.equal(y_pred, y_pred_max)
    for c_p, c_t in product(range(nb_cl), range(nb_cl)):
        final_mask += (K.cast(weights[c_t, c_p],K.floatx()) * K.cast(y_pred_max_mat[:, c_p] ,K.floatx())* K.cast(y_true[:, c_t],K.floatx()))
    return K.categorical_crossentropy(y_true, y_pred) * final_mask

#########################

def categorical_crossentropy_w_wrap(weights):
# https://datascience.stackexchange.com/questions/41698/how-to-apply-class-weight-to-a-multi-output-model
# weights[i, j] defines the weight for an example of class i which was falsely classified as class j.
    
    #weights = K.variable(weights)
    
    def loss(y_true, y_pred):
        y_true = y_true [:,2:]
        nb_cl = len(weights)
        nb_cl = y_true.shape[1]
        final_mask = K.zeros_like(y_pred[:, 0])
        y_pred_max = K.max(y_pred, axis=1)
        y_pred_max = K.reshape(y_pred_max, (K.shape(y_pred)[0], 1))
        y_pred_max_mat = K.cast(K.equal(y_pred, y_pred_max), K.floatx())
        for c_p, c_t in product(range(nb_cl), range(nb_cl)):
            final_mask += (weights[c_t, c_p] * y_pred_max_mat[:, c_p] * y_true[:, c_t])
        return K.categorical_crossentropy(y_true, y_pred) * final_mask

    return loss


def weighted_categorical_crossentropy_2(y_true, y_pred):
    """
    A weighted version of keras.objectives.categorical_crossentropy
    
    Variables:
        weights: numpy array of shape (C,) where C is the number of classes
    
    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
        
    https://gist.github.com/wassname/ce364fddfc8a025bfab4348cf5de852d
    """
    # y_true = tensor with weights as first column
    # y_pred = tensor with pediction after softmax
    
    y_true_val = y_true [:,1:]
    # extract weights and transform vector of len BATCH _SIZE in array of size 1 x BATCH_SIZE
    weights = y_true[:,0].reshape(-1, 1)
    # transform weight in matrix of size NUM_CLASS x BATCH_SIZE
    weights_m = np.repeat(weights, y_true_val.shape[1], axis = 1)
        
    # scale predictions so that the class probas of each sample sum to 1
    y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
    # clip to prevent NaN's and Inf's
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    # calc the loss by multiplying by the weights matrix
    # correct and uncorrect prediction are both weighted
    loss = y_true_val * K.log(y_pred)  * weights_m
    loss = -K.sum(loss, -1)
    return loss


def categorical_crossentropy_weighted(y_true, y_pred):
    weights = y_true[:,0]    
    y_true_val = y_true [:,1:]
    return weights * categorical_crossentropy(y_true_val, y_pred)     

def mae_tol_param(tol):
    '''Calculates mean average error with a tolerance of tol 
        hence if absolute delta <= tol, then error si considered as null.
    '''
    def mae_tol(y_true,y_pred):
        return K.mean( K.maximum(K.abs(y_true - y_pred) - tol, 0) )
    return mae_tol

def precision(y_true, y_pred):
    '''Calculates the precision, a metric for multi-label classification of
    how many selected items are relevant.
    '''
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):
    '''Calculates the recall, a metric for multi-label classification of
    how many relevant items are selected.
    '''
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def fbeta(y_true, y_pred, beta=1):
    '''Calculates the F score, the weighted harmonic mean of precision and recall.
    This is useful for multi-label classification, where input samples can be
    classified as sets of labels. By only using accuracy (precision) a model
    would achieve a perfect score by simply assigning every class to every
    input. In order to avoid this, a metric should penalize incorrect class
    assignments as well (recall). The F-beta score (ranged from 0.0 to 1.0)
    computes this, as a weighted mean of the proportion of correct class
    assignments vs. the proportion of incorrect class assignments.
    With beta = 1, this is equivalent to a F-measure. With beta < 1, assigning
    correct classes becomes more important, and with beta > 1 the metric is
    instead weighted towards penalizing incorrect class assignments.
    '''
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')
        
    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score

def fbeta_2(y_true, y_pred, threshold_shift=0):
    beta = 1

    # just in case of hipster activation at the final layer
    y_pred = K.clip(y_pred, 0, 1)

    # shifting the prediction threshold from .5 if needed
    y_pred_bin = K.round(y_pred + threshold_shift)

    tp = K.sum(K.round(y_true * y_pred_bin)) + K.epsilon()
    fp = K.sum(K.round(K.clip(y_pred_bin - y_true, 0, 1)))
    fn = K.sum(K.round(K.clip(y_true - y_pred, 0, 1)))

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    beta_squared = beta ** 2
    return (beta_squared + 1) * (precision * recall) / (beta_squared * precision + recall + K.epsilon())



def weighted_categorical_crossentropy_wrap(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy
    
    Variables:
        weights: numpy array of shape (C,) where C is the number of classes
    
    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
        
    https://gist.github.com/wassname/ce364fddfc8a025bfab4348cf5de852d
    """
    
    weights = K.variable(weights)
        
    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss
    
    return loss


# ********************* OTHER CUSTOM LOSS *********************

#def custom_loss(y_true, y_pred)
#    weights = y_true[:,1]
#    y_true = y_true [:,0]

# ********************* TESTS *********************

# ============================================================================= 
def tests():
    
    import numpy as np
    from keras.activations import softmax
    from keras.objectives import categorical_crossentropy
    
    # 1. weights = 1
    y_true = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1], [1, 0, 0]])
    # y_true_ includes the weights / all weights to 1 in y_true_w
    y_true_w = np.array([[1,1, 0, 1, 0], [1,1, 1, 0, 0], [1,1, 0, 0, 1], [1,1, 1, 0, 0]])

    # second is incorrect
    y_pred = np.array([[0, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]])
    # softmax version to feed in categorical_crossentropy
    y_pred = K.variable(y_pred)
    y_pred_s = softmax(y_pred)
    y_pred_s_eval = y_pred_s.eval(session=K.get_session())
    
    loss_abs1 = categorical_crossentropy(y_true,y_pred_s).eval(session=K.get_session())
    loss_abs2 = categorical_crossentropy_abs(y_true_w,y_pred_s).eval(session=K.get_session())
    np.testing.assert_almost_equal(loss_abs1,loss_abs2)
    
    loss_abs3 = categorical_focal_loss(gamma=0, alpha=1)(y_true_w,y_pred_s).eval(session=K.get_session())
    np.testing.assert_almost_equal(loss_abs1,loss_abs3)
    
    artist_weights_matrix = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    loss_artits_wrap = categorical_crossentropy_w_wrap(artist_weights_matrix)
    loss_w1a = loss_artits_wrap(y_true_w,y_pred_s).eval(session=K.get_session())
    np.testing.assert_almost_equal(loss_abs1,loss_w1a)

    loss_artist_wb = lambda y_true, y_pred: w_categorical_crossentropy(y_true, y_pred, weights=artist_weights_matrix)
    loss_w1b = loss_artist_wb(y_true_w,y_pred_s).eval(session=K.get_session())
    np.testing.assert_almost_equal(loss_abs1,loss_w1b)

    loss_w1c = w_categorical_focal_loss(gamma=0, alpha=1)(y_true_w,y_pred_s).eval(session=K.get_session())
    np.testing.assert_almost_equal(loss_w1b,loss_w1c)
    
    
    # 2. weights != 1 ==> = 2 for 1st artist
    # change ground truth and weights
    # 1 error at secon line + 1 at last line
    y_true_w = np.array([[1,1, 0, 1, 0], [2,2, 1, 0, 0], [1,1, 0, 0, 1], [1,1, 0, 1, 0]])
    artist_weights_matrix_2 = np.array([[1, 2, 2], [1, 1, 1], [1, 1, 1]])
    loss_artits_w_2 = categorical_crossentropy_w_wrap(artist_weights_matrix_2)
    loss_w2a = loss_artits_w_2(y_true_w,y_pred_s).eval(session=K.get_session())
    loss_abs2 = categorical_crossentropy_abs(y_true_w,y_pred_s).eval(session=K.get_session())
    
    loss_artits_wb = lambda y_true, y_pred: w_categorical_crossentropy(y_true, y_pred, weights=artist_weights_matrix_2)
    loss_w2b = loss_artits_wb(y_true_w,y_pred_s).eval(session=K.get_session())
    np.testing.assert_almost_equal(loss_w2a,loss_w2b)
    
    loss_w2c = w_categorical_focal_loss(gamma=0, alpha=1)(y_true_w,y_pred_s).eval(session=K.get_session())
    np.testing.assert_almost_equal(loss_w2b,loss_w2c)

    # non-weighted focal loss with default param
    loss_w2d = categorical_focal_loss(gamma=2, alpha=.25)(y_true_w,y_pred_s).eval(session=K.get_session())

    # weighted focal loss with default param
    loss_w2e = w_categorical_focal_loss(gamma=2, alpha=.25)(y_true_w,y_pred_s).eval(session=K.get_session())
    print('done')
    
#tests()