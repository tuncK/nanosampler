#!/usr/bin/env python

# Implementation of a custom-made subsampling layer
# We tried various options

from keras import backend
from keras.layers import Dense, Dropout, Input, Masking
from keras.models import Model
import numpy as np
import tensorflow as tf


class SubsamplingLayer(tf.keras.layers.Layer):
    def __init__(self, dim, num_measurements, **kwargs):
        super().__init__(**kwargs)
        self.num_measurements = num_measurements
        initialiser = tf.keras.initializers.GlorotNormal()
        
        # Stored vector array of learnable weights attributed to each feature.
        # Have to stick to 'kernel' misnomer to match other Keras layers.
        self.kernel = tf.Variable(shape=(dim,), initial_value=initialiser(shape=(dim,)), trainable=True, dtype='float32', name='weights')

    @staticmethod
    def get_topk_mask(x, k):
        top_k_vals = tf.math.top_k(x, k=k).values
        threshold = tf.math.reduce_min(top_k_vals)
        
        # Only the highest scoring features will be retained, all others 0.
        mask = tf.where(x >= threshold, 1.0, 0.0) # 1 0
        return mask

    def call(self, inputs):
        mask = self.get_topk_mask(inputs, self.num_measurements)
        return mask * inputs


# Tailor-made regulariser to heavily penalise measuring more and more proteins.
class L0Regularizer(tf.keras.regularizers.Regularizer):
    """
    Constrains weight tensors such that among the incoming edges, only 1 of them is 1, else 0.
    This effectively selects only 1 of those nodes as input, filtering out the rest.
    """
    def __init__(self, l0=0.0, steepness=1.0):
        self.l0 = l0
        self.steepness = steepness
    
    def __call__(self, w):
        # In the ideal case, only 1 incoming edge is supposed to be 1, all others 0.
        penalty = tf.math.reduce_sum(tf.math.abs(tf.math.reduce_sum(w, axis=0) - 1))
        return self.l0 * tf.cast(penalty, tf.float32)

    def get_config(self):
        return {'l0': float(self.l0)}


class constrain_01(tf.keras.constraints.Constraint):
    """
    Constrains the weights of the subsampling layer so that the weights
    are in [0,1] or {0,1}.
    """
    def __init__(self, steepness=1.0):
        self.steepness = steepness

    def __call__(self, w):
        #ws = tf.math.sigmoid(w * self.steepness)
        #ww = tf.cast(tf.math.greater_equal(w, 0.0), w.dtype)
        #ww = tf.cast(tf.math.less_equal(w, 1.0), w.dtype)
        
        w_min = tf.math.reduce_min(w)
        w_max = tf.math.reduce_max(w)
        w_scaled = (w - w_min) / (w_max - w_min)
        
        return w_scaled


def subsampling_classifier(dims, num_measurements, latent_act='relu', init='he_normal', dropout_rate=0.0):
    # init='glorot_uniform' was worse in loss during testing
    
    """
    Fully connected auto-encoder model, symmetric. SAE or DAE.

    Parameters
    ----------
    dims : int array
        List of number of units in each layer of encoder. dims[0] is input dim, dims[-1] is the hidden layer.
        The decoder is symmetric with the encoder. So the total number of layers of the
        auto-encoder is 2*len(dims)-1.

    act : str
        Activation function, if any, to put in the internal dense layers

    Returns
    ----------
    (ae_model, encoder_model) : Model of the autoencoder and model of the encoder
    """

    # input layer
    input_layer = Input(shape=dims[0], name='input')
    x = input_layer

    # Ignore all but num_measurements many proteins, i.e. mimic limited experimental sampling
    #x = SubsamplingLayer(dims[0], num_measurements=num_measurements, name='target_selection')(x)

    # Soft alternative to above
    x = Dense(num_measurements, name='target_selection', use_bias=False, kernel_initializer=init, kernel_constraint=constrain_01(), kernel_regularizer=L0Regularizer(l0=0))(x)

    dense_layer_idx = 1
    for layer_dim in dims[1:]:
        x = Dense(layer_dim, activation=latent_act, kernel_initializer=init, name='dense_%d' % dense_layer_idx)(x)
        x = Dropout(rate=dropout_rate, name='dropout_%d' % dense_layer_idx)(x)
        dense_layer_idx+=1
    
    x = Dense(1, activation='sigmoid', kernel_initializer=init, name='dense_%d' % dense_layer_idx)(x)
    
    output_layer = x
    model = tf.keras.Model(input_layer, output_layer)
    model.summary()
    return model
