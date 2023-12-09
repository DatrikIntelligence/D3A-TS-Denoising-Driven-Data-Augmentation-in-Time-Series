import tensorflow as tf
from collections import Counter
import pandas as pd
import numpy as np
import random 
import pickle as pk
import gc
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import logging


FEATURE_NAMES = ['acc']


def load_file(data_file, target_file):

    data = open(data_file, 'r').readlines()
    target = open(target_file, 'r').readlines()
    X = np.zeros((len(data), 1, 561))
    Y = np.zeros((len(data,)))
    for i, (signal, y) in enumerate(zip(data, target)):
        signal = list(filter(None, signal.split(' ')))
        label = int(float(y))
        signal = np.array([float(d) for d in signal])
        X[i, 0] = signal
        Y[i] = label-1
        
    return X,Y

def load_data(return_test=True, return_train=True):
    if return_train:
        X_train, Y_train = load_file('../data/human-activity/train/X_train.txt', 
                                     '../data/human-activity/train/y_train.txt')
        scaler = StandardScaler()
        X_train[:, 0] = scaler.fit_transform(X_train[:, 0]).round(3)
        X_train = (X_train, Y_train)
        
    if return_test:
        X_test, Y_test = load_file('../data/human-activity/test/X_test.txt', 
                                   '../data/human-activity/test/y_test.txt')
        scaler = StandardScaler()
        X_test[:, 0] = scaler.fit_transform(X_test[:, 0]).round(3)
        X_test = (X_test, Y_test)
    
    logging.info("Readed dataset")
    
    np.random.seed(0)

    if return_test and return_train:
        return X_train, X_test
    elif return_test:
        return None, X_test
    else:
        return X_train, None

    
def load_generators(return_test=True, return_train=True, tslen=140):
    X_train, X_test = load_data(return_test, return_train)

    gen_train, gen_test = None, None
    
    if return_train:
        logging.info("Creating train generator")
        gen_train = HASequence(X_train, 
                                extra_channel=False, batch_size=32, 
                                batches_per_epoch=500, ts_len=tslen) 
        
        del X_train
        gc.collect()

    if return_test:
        logging.info("Creating test generator")
        gen_test = HASequence(X_test, 
                                extra_channel=False, batch_size=32, 
                                batches_per_epoch=500, ts_len=tslen) 
        del X_test
        gc.collect()
    
    return gen_train, gen_test
    
def compute_baseline(Y_train, Y_test):
    y_mean, y_std = Y_train.mean(), Y_train.std()
    y_test = np.array(Y_test).reshape((Y_test.shape[0],1)).astype('float')
    
    pred_label = int(y_mean)
    y_pred = np.zeros((Y_test.shape[0], 5))
    y_pred[:, pred_label] = 1.0
    
    print(y_test.shape, y_pred.shape)

    #f1score = tf.keras.metrics.F1Score(name='f1-score')(Y_test, y_pred).numpy()
    acurracy = tf.keras.metrics.SparseCategoricalAccuracy()
    acurracy.update_state(y_test, y_pred)
    acurracy = acurracy.result().numpy()
    logloss = tf.keras.metrics.SparseCategoricalCrossentropy()
    logloss.update_state(y_test, y_pred)
    logloss = logloss.result().numpy()
    
    logging.info(f"Accuracys: {acurracy}")
    logging.info(f"Logloss: {logloss}")
    
    return {'acurracy': acurracy, 'logloss': logloss}

class HASequence(tf.keras.utils.Sequence):

    def __init__(self, X, ts_len=561, batch_size=32, 
                 batches_per_epoch=100, extra_channel=False,
                 output_extra_channel=False, return_label=True):
        self.batch_size = batch_size
        self.return_label = return_label
        self.output_extra_channel = output_extra_channel
        self.ts_len = ts_len
        self.attributes = FEATURE_NAMES
        self.batches_per_epoch = batches_per_epoch
        self._X, self._Y  = X
        self._ids = np.arange(self._X.shape[0])
        self.extra_channel = extra_channel

    def __len__(self):
        """Denotes the number of batches per epoch
        :return: number of batches per epoch
        """
        return self.batches_per_epoch
    
    def __getitem__(self, index):
        """Generate one batch of data
        :param index: index of the batch
        :return: X and y when fitting. X only when predicting
        """
        X = self._X
        _X = []
        _y = []
        for _ in range(self.batch_size):
            sid = random.choice(self._ids)
            signal = self._X[sid]
            nrows = signal.shape[1]
            cut = random.randint(0, nrows - self.ts_len)
            s = signal[:, cut: cut + self.ts_len]
            y = self._Y[sid]
            _X.append(s)
            _y.append(y)

        
        _X = np.array(_X)
        if self.extra_channel:
            _X = _X.reshape(_X.shape + (1,))

       
        return _X, np.array(_y)
        
    def on_epoch_end(self):
        """Updates indexes after each epoch
        """
        pass
