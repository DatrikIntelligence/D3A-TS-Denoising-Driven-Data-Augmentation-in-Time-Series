import tensorflow as tf
import pandas as pd
import numpy as np
import random 
import logging
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import gc

FEATURE_NAMES = ["V_acc", "H_acc"]


def load_data(return_test=True, return_train=True, train_bearings=[1, 3, 6, 7]):

    X_train = pd.read_csv('../data/pronostia/pronostia_train.csv')
    X_test = pd.read_csv('../data/pronostia/pronostia_test.csv')
    X = pd.concat((X_train,X_test), axis=0)
    
    scaler = StandardScaler()
    X.loc[:, FEATURE_NAMES] = scaler.fit_transform(X.loc[:,FEATURE_NAMES]).round(3)
    
    test_bearings = [i for i in range(1, 8) if i not in train_bearings]
    X_test = X[X.Bearing.isin(test_bearings)]
    X_train = X[X.Bearing.isin(train_bearings)]
    
    

    return X_train, X_test

def load_generators(return_test=True, return_train=True, tslen=128):
    X_train, X_test = load_data(return_test, return_train)

    gen_train, gen_test = None, None
    
    if return_train:
        logging.info("Creating train generator")
        gen_train = PRONOSTIASequence(X_train, 
                                extra_channel=False, batch_size=32, 
                                batches_per_epoch=10000, ts_len=tslen) 
        
        del X_train
        gc.collect()

    if return_test:
        logging.info("Creating test generator")
        gen_test = PRONOSTIASequence(X_test, 
                                extra_channel=False, batch_size=32, 
                                batches_per_epoch=5000, ts_len=tslen) 
        del X_test
        gc.collect()
    
    return gen_train, gen_test
    
def compute_baseline(Y_train, Y_test):
    y_mean, y_std = Y_train.mean(), Y_train.std()
    y_pred = [y_mean] * Y_test.shape[0]
    
    logging.info(f"Mean prediction: {y_mean}")

    mae = tf.keras.metrics.MeanAbsoluteError(name='mae')(Y_test, y_pred).numpy()
    mse = tf.keras.metrics.MeanSquaredError(name='mse')(Y_test, y_pred).numpy()
    
    logging.info(f"MAE: {mae}")
    logging.info(f"MSE: {mse}")
    
    return {'mae': mae, 'mse': mse}



class PRONOSTIASequence(tf.keras.utils.Sequence):

    def __init__(self, data, batches_per_epoch=1000, batch_size=32, split_channel=False, 
                 extra_channel=True, ts_len=256):
        self.data = data
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch
        self.bearings = self.data[['Condition', 'Bearing']].drop_duplicates().values
        logging.info(f"Pronostia bearings {str(self.bearings)}. Total bearings: {self.bearings.shape[0]}")
        self.data = {}
        self.rul_max = {}
        self.split_channel = split_channel
        self.extra_channel = extra_channel
        self.__num_samples = data.shape[0]
        self.ts_len = ts_len
        D = data
        for cond, bearing in self.bearings:
            d = D[(D.Condition == cond) & (D.Bearing==bearing)]
            self.rul_max[(cond, bearing)] = d.RUL.max()
            self.data[(cond, bearing)] = d[['V_acc', 'H_acc', 'RUL']].values
            
    def __len__(self):
        return self.batches_per_epoch

    def __getitem__(self, idx):
        D = self.data
        ts_len = self.ts_len
        if self.split_channel:
            X = np.zeros(shape=(self.batch_size, 2, ts_len//2, 2))
            Y = np.zeros(shape=(self.batch_size,))
            for i in range(self.batch_size):
                cond, bearing = self.bearings[random.randint(0, self.bearings.shape[0]-1)]
                Db = self.data[(cond, bearing)]
                L = (Db.shape[0] // ts_len) 

                k = random.randint(0, L-2) * ts_len

                l = ts_len//2
                X[i, :, :, 0] = Db[k:k+l, :2].T
                X[i, :, :, 1] = Db[k+l:k+2*l, :2].T
                Y[i] = Db[k+1:k+2*l, 2][-1] / self.rul_max[(cond, bearing)]   
        else:
            if self.extra_channel:
                X = np.zeros(shape=(self.batch_size, 2, ts_len, 1))
            else:
                X = np.zeros(shape=(self.batch_size, 2, ts_len))
                
            Y = np.zeros(shape=(self.batch_size,))
            for i in range(self.batch_size):
                cond, bearing = self.bearings[random.randint(0, self.bearings.shape[0]-1)]
                Db = self.data[(cond, bearing)]
                L = (Db.shape[0] // ts_len) 
                k = random.randint(0, L-2) * ts_len
                if self.extra_channel:
                    X[i, :, :, 0] = Db[k:k+ts_len, :2].T
                else:
                    X[i, :, :] = Db[k:k+ts_len, :2].T
                Y[i] = Db[k:k+ts_len, 2][-1] / self.rul_max[(cond, bearing)]  
        
        return X, Y
