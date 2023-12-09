import tensorflow as tf
from collections import Counter
import pandas as pd
import numpy as np
import random 
import pickle as pk
import gc
from sklearn.preprocessing import MinMaxScaler
import logging


FEATURE_NAMES = ['alt', 'Mach', 'TRA', 'T2', 'T24', 'T30', 'T48', 'T50', 'P15', 'P2',
           'P21', 'P24', 'Ps30', 'P40', 'P50', 'Nf', 'Nc', 'Wf', 'Fc', 'hs']



def load_data(return_test=True, return_train=True):
    X = pd.read_hdf('../data/ncmapss/full_dataset.h5', key='phm')
    logging.info("Readed full dataset")
    
    np.random.seed(0)
    ids = X.id.unique()
    np.random.shuffle(ids)
    
    scaler = MinMaxScaler()
    
    if return_train:
        X_train = X[X.id.isin(ids[:int(len(ids)*0.7)])]
        X_train.loc[:,FEATURE_NAMES] = scaler.fit_transform(X_train.loc[:,FEATURE_NAMES]).round(3)
        logging.info("Splitted train")
        
    if return_test:
        X_test = X[X.id.isin(ids[int(len(ids)*0.7):])]
        X_test.loc[:,FEATURE_NAMES] = scaler.fit_transform(X_test.loc[:,FEATURE_NAMES]).round(3)
        logging.info("Splitted test")
        
    del X
    gc.collect()
    
    if return_test and return_train:
        return X_train, X_test
    elif return_test:
        return None, X_test
    else:
        return X_train, None

    
def load_generators(return_test=True, return_train=True, tslen=128):
    X_train, X_test = load_data(return_test, return_train)

    gen_train, gen_test = None, None
    
    if return_train:
        logging.info("Creating train generator")
        gen_train = NCMAPSSSequence(X_train, 
                                extra_channel=False, batch_size=32, 
                                batches_per_epoch=500, ts_len=tslen) 
        
        del X_train
        gc.collect()

    if return_test:
        logging.info("Creating test generator")
        gen_test = NCMAPSSSequence(X_test, 
                                extra_channel=False, batch_size=32, 
                                batches_per_epoch=200, ts_len=tslen) 
        del X_test
        gc.collect()
    
    return gen_train, gen_test
    
def compute_baseline(Y_train, Y_test):
    y_mean, y_std = Y_train.mean(), Y_train.std()
    y_pred = [y_mean] * Y_test.shape[0] 

    mae = tf.keras.metrics.MeanAbsoluteError(name='mae')(Y_test, y_pred).numpy()
    mse = tf.keras.metrics.MeanSquaredError(name='mse')(Y_test, y_pred).numpy()
    
    logging.info(f"MAE: {mae}")
    logging.info(f"MSE: {mse}")
    
    return {'mae': mae, 'mse': mse}

class NCMAPSSSequence(tf.keras.utils.Sequence):

    def __init__(self, X, ts_len=10, batch_size=32, 
                 batches_per_epoch=100, extra_channel=False,
                 output_extra_channel=False, return_label=True):
        self.batch_size = batch_size
        self.return_label = return_label
        self.output_extra_channel = output_extra_channel
        self.ts_len = ts_len
        self.attributes = FEATURE_NAMES
        self.batches_per_epoch = batches_per_epoch
        self._X = {}
        self._Y = {}
        self._ids = X.id.unique()
        self.extra_channel = extra_channel
        for _id in self._ids:
            self._X[_id] = X.loc[(X.id==_id), self.attributes].values
            self._Y[_id] = X.loc[(X.id==_id), 'Y'].values


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
            unit = self._X[sid]
            nrows = unit.shape[0]
            cut = random.randint(0, nrows - self.ts_len)
            s = unit[cut: cut + self.ts_len].T
            y =self._Y[sid][cut + self.ts_len-1]
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
