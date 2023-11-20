#!/usr/bin/env python

# Model for Nanodiag project which subsamples the input nodes
# and then tries to learn the labels with MLP.

from keras import backend
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, LambdaCallback
from keras.layers import Dense, Dropout, Input, Masking
from keras.models import Model
from keras.optimizers import Adam
from tensorflow.keras.utils import plot_model

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import time


# Fix a np.random.seed for reproducibility in numpy processing
np.random.seed(42)


class filterInputs(tf.keras.constraints.Constraint):
  """
  Constrains weight tensors such that among the incoming edges, only 1 of them is 1, else 0.
  This effectively selects only 1 of those nodes as input, filtering out the rest.
  """

  def __init__(self):
    self.ref_value = 1

  def __call__(self, w):
    mean = tf.reduce_mean(w)
    return w - mean + self.ref_value

  def get_config(self):
    return {'ref_value': self.ref_value}



class SubsamplingLayer(tf.keras.layers.Layer):
    def __init__(self, dim, num_measurements, **kwargs):
        super().__init__(**kwargs)
        self.w = tf.Variable(shape=(dim), initial_value=np.ones((dim)), trainable=True, dtype='float32', name='weights', constraint=tf.keras.constraints.unit_norm(axis=0))
        self.num_measurements = num_measurements

    def call(self, inputs):
        top_k_vals = tf.math.top_k(self.w, k=self.num_measurements).values
        threshold =  tf.math.reduce_min(top_k_vals)
        
        # Option 1: set unselected antibody targets to 0.       
        masked = tf.where(self.w-threshold >= 0, inputs, 0)

        # Option 2: mask unselected antibodies (i.e. -1)
        #masked = tf.where(self.w-threshold >= 0, inputs, -1)
        #masked = Masking(mask_value=-1)(masked)
        
        return masked


# Tailor-made regulariser to heavily penalise measuring more and more proteins.
# @tf.keras.utils.register_keras_serializable(package='Custom', name='l0')
class L0Regularizer(tf.keras.regularizers.Regularizer):
    def __init__(self, l0=0.0, steepness=1.0):
        self.l0 = l0
        self.steepness = steepness
    
    def __call__(self, w):
        # In the ideal case, only 1 incoming edge is supposed to be 1, all others 0.
        wp = tf.math.ceil(tf.math.sigmoid(w * self.steepness) - 0.5)
        penalty = tf.cast(tf.math.reduce_sum(tf.math.count_nonzero(wp, axis=0) - 1), tf.float32)
        return self.l0 * penalty
        #return self.l0 * tf.math.reduce_sum(tf.math.abs( tf.math.reduce_sum(tf.cast(x>0.2, tf.float32), axis=1) - 1))

    def get_config(self):
        return {'l0': float(self.l0)}


class constrain01(tf.keras.constraints.Constraint):
    """
    Constrains the weights of the subsampling layer so that the weights
    are in [0,1] or {0,1}.
    """
    def __init__(self, steepness=1.0):
        self.steepness = steepness

    def __call__(self, w):
        # return w
        wp = tf.math.sigmoid(w * self.steepness)
        return tf.math.ceil(wp-0.5)
        #wp = tf.cast(tf.greater_equal(w, 0.0), tf.float32)
        #return wp / (backend.epsilon() + tf.reduce_max(wp, axis=0, keepdims=True))


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
    input_layer = Input(shape=(dims[0]), name='input')
    x = input_layer

    # Ignore all but num_measurements many proteins, i.e. mimic limited experimental sampling
    # x = SubsamplingLayer(dims[0], num_measurements=num_measurements, name='target_selection')(x)

    # Soft alternative to above
    x = Dense(num_measurements, name='dense_0', kernel_constraint=constrain01())(x) 
    # , kernel_regularizer=L0Regularizer(l0=0.1) )(x)

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


class Modality(object):
    """
    Auto-encoder object for a stand-alone data modality

    Parameters
    ----------
    data : str
        Filename containing the entire training dataset
        File should contain an ndarray of shape (`n_features`,`n_samples`)

    clipnorm_lim : float
        Threshold for gradient normalisation for numerical stability during training.
        If the norm of the gradient exceeds this threshold, it will be scaled down to this value.
        The lower is the more stable, at the expense of increased the training duration.

    max_training_duration : int
        Maximum duration to be allowed for the AE training step to take. Walltime measured in seconds,
        by default the training duration is unlimited.

    seed : int
        Seed for the number random generator. Defaults to 0.
    """

    def __init__(self, Xfile, Yfile, clipnorm_lim=1, seed=0, max_training_duration=np.inf, **kwargs):
        self.t_start = time.time()
        self.Xfilename = str(Xfile)
        self.Yfilename = str(Yfile)      
        self.seed = seed
        self.max_training_duration = max_training_duration
        self.prefix = ''
        self.clipnorm_lim = clipnorm_lim
        self.dataset_ids = None

        dataset_name = self.Xfilename.split('/')[-1].split('_')[-1].split('.')[0]
        self.modelName = dataset_name + '_' + str(self.seed) + '_' + str(clipnorm_lim)

        # Create the output (sub)folder if necessary
        self.output_dir = './results/'
        os.makedirs(self.output_dir, exist_ok=True)


    def load_X_data(self, dtype=None):
        """
            Read training data (X) from file
        """

        # read file
        if os.path.isfile(self.Xfilename):
            raw = pd.read_csv(self.Xfilename, sep='\t', index_col=0, header=0)
        else:
            raise FileNotFoundError("File {} does not exist".format(self.Xfilename))

        # Keep the patient ids etc. to be able to match to labels later on.
        # We will remove pandas auto-added suffixes on duplicates
        # ABC, ABC.1, ABC.2 ... -> ABC
        self.dataset_ids = [x.split('.')[0] for x in list(raw)]

        # load data
        # self.X_train = raw.transpose().values.astype('int32')
        self.X_train = raw.transpose().values.astype(dtype)

        # put nothing or zeros for y_train, y_test, and X_test, at least temporarily
        self.y_train = np.zeros(shape=(self.X_train.shape[0])).astype(dtype)
        self.X_test = np.zeros(shape=(1, self.X_train.shape[1])).astype(dtype)
        self.y_test = np.zeros(shape=(1,)).astype(dtype)


    def load_Y_data(self, dtype=None):
        """
        Reads class labels (Y) from file
        """

        if os.path.isfile(self.Yfilename):
            imported_labels = pd.read_csv(self.Yfilename, sep='\t', index_col=0, header=0)

            # There might be duplicate measurements for the same patient.
            # i.e., some patient identifiers might need to be repeated.
            # The order of params also needs to match to the training data X
            labels = imported_labels.loc[self.dataset_ids]
        else:
            raise FileNotFoundError("{} does not exist".format(self.Yfilename))

        # Label data validity check
        if not labels.values.shape[1] > 1:
            label_flatten = labels.values.reshape((labels.values.shape[0])).astype(dtype)
        else:
            raise IndexError('The label file contains more than 1 column.')

        # train and test split
        split_data = train_test_split(self.X_train, label_flatten, test_size=0.2, random_state=self.seed, stratify=label_flatten)
        self.X_train, self.X_test, self.y_train, self.y_test = split_data
        self.printDataShapes()

    # Custom keras callback function to limit total training time
    # This is needed for early stopping the procedure during BOHB
    class TimeLimit_Callback(Callback):
        def __init__(self, verbose=False, max_training_duration=np.inf):
            self.training_start_time = time.time()
            self.verbose = verbose
            self.max_training_duration = max_training_duration

        def on_epoch_end(self, epoch, logs={}):
            duration = time.time() - self.training_start_time
            if self.verbose:
                print('%ds passed so far' % duration)

            if duration >= self.max_training_duration:
                print('Training exceeded time limit (max=%ds), stopping...'
                      % self.max_training_duration)
                self.model.stop_training = True
                self.stopped_epoch = epoch

    def train_classifier(self, latent_dims, epochs=1000, batch_size=1000, verbose=2, loss='binary_crossentropy', act='relu', patience=20, val_rate=0.2, plot_progress=True, save_model=False, dropout_rate=0.0, **kwargs):
        # Generate an experiment identifier string for the output files
        self.prefix = 'p' + str(patience) + '_'

        # callbacks for each epoch
        callbacks = self.set_callbacks(patience=patience, save_model=save_model)

        # insert input shape into dimension list
        latent_dims.insert(0, self.X_train.shape[1])

        # create classifier model
        self.classifier = subsampling_classifier(dims=latent_dims, num_measurements=100, latent_act=act, dropout_rate=dropout_rate)
       
        # compile the model
        customised_adam = Adam(clipnorm=self.clipnorm_lim)
        self.classifier.compile(optimizer=customised_adam, loss=loss, metrics=['binary_accuracy'])

        # Start training procedure
        self.history = self.classifier.fit(x=self.X_train, y=self.y_train, epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=verbose, validation_data=(self.X_test, self.y_test))

        if plot_progress:
            self.saveLossProgress()
            plot_model(self.classifier, 'model.png', show_shapes=True)


    def set_callbacks(self, patience, save_model):
        callbacks = [EarlyStopping(monitor='val_loss', patience=patience, mode='min', verbose=0, restore_best_weights=True),
                     self.TimeLimit_Callback(max_training_duration=self.max_training_duration)]
                   
        #callbacks = [EarlyStopping(monitor='val_binary_accuracy', patience=patience, mode='max', verbose=0),
        #             self.TimeLimit_Callback(max_training_duration=self.max_training_duration)]

        # Exports the model to file at each iteration.
        # Due to early stopping, the final model is not necessarily the best model.
        # Constant disk IO may slow down the training considerably.
        if save_model:
            self.model_out_file = self.output_dir + '/' + self.modelName + '.h5'
            model_write_callback = ModelCheckpoint(self.model_out_file, monitor='val_loss', mode='min', verbose=1, save_best_only=True, save_weights_only=True),
            callbacks.append(model_write_callback)

            # clean up model checkpoint before use
            if os.path.isfile(self.model_out_file):
                os.remove(self.model_out_file)

        return callbacks


    def printDataShapes(self, train_only=False):
        print("X_train.shape: ", self.X_train.shape)
        print("y_train.shape: ", self.y_train.shape)
        print("X_test.shape: ", self.X_test.shape)
        print("y_test.shape: ", self.y_test.shape)


    # ploting loss progress over epochs
    def saveLossProgress(self):
        plt.rcParams.update({'font.size': 14})
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])

        plt.title('Autoencoder model loss')
        plt.ylabel('Binary cross entropy')
        plt.xlabel('Epoch')
        plt.legend(['train', 'val.'], loc='upper right')
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig('loss_curves.png')
        plt.close()


def train_modality(Xfile, Yfile, gradient_threshold=1, latent_dims=[16,8], max_training_duration=np.inf, seed=42, dropout_rate=0.0):
    """
    A function that trains the AE only, but not the classifier itself.

    Currently this is invoked by BOHB for parameter optimisation.

    Parameters
    ----------
    Xfile : str
        Name of the file containing features matrix (i.e. X). First row and column
        will be treated as labels and ignored. REQUIRED

    Yfile : str
        Name of the file containing the class labels. Should be a 2-column tsv.
        1st column should contain the sample identifiers, and should match those in
        Xfile. 2nd column should contain binary class of each sample. REQUIRED

    gradient_threshold : float
        Maximum gradient magnitude to be allowed during training. Smaller values
        reduce the likelihood of gradient explosion, however might considerably
        increase the training time. Not used by default.

    latent_dims : int or array of ints
        Number of dimensions of the latent space. 8 by default and only used by
        AE and VAE. To include hidden layer(s), set latent_dims to a list of ints

    num_filters : int
        Number of kernels to be trained. CAE only, 3 by default.

    max_training_duration : float
        Maximum duration to allow during the AE training, in seconds. If none
        provided, there is no time limit.

    seed : int
        Seed for the random number generator. Defaults to 42.
    """

    # create an object and load data
    # Each different experimental component needs to be treated as 1 separate Modality.
    m = Modality(Xfile=Xfile, Yfile=Yfile, seed=seed, clipnorm_lim=gradient_threshold, max_training_duration=max_training_duration)

    # load data into the object
    m.load_X_data()
    m.load_Y_data()

    # time check after data has been loaded
    m.t_start = time.time()

    m.train_classifier(latent_dims=latent_dims, dropout_rate=dropout_rate)
    return


if __name__ == '__main__':
    dataset = 'gdac' # gdac or demo
    train_modality(Xfile='./input/%s_X.tsv' % dataset, Yfile='./input/%s_y.tsv' % dataset, latent_dims=[100, 10], dropout_rate=0.0, gradient_threshold=0.1)


