from collections import defaultdict
import tensorflow as tf
import random
import numpy as np 
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from collections import Counter
import h5py

class DataGenerator(tf.keras.utils.Sequence):
    """
    """
    def __init__(self, X, attributes, window_size=10, batch_size=32, 
                 noise_level=0, epoch_len_reducer=100, add_extra_channel=False,
                 output_extra_channel=False, return_label=True):
        self.batch_size = batch_size
        self.return_label = return_label
        self.output_extra_channel = output_extra_channel
        self.window_size = window_size
        self.noise_level = noise_level
        self.attributes = attributes
        self.epoch_len_reducer = epoch_len_reducer
        self._X = {}
        self._Y = {}
        self._ids = X.id.unique()
        self.add_extra_channel = add_extra_channel
        for _id in self._ids:
            self._X[_id] = X.loc[(X.id==_id), self.attributes].values
            self._Y[_id] = X.loc[(X.id==_id), 'Y'].values
        self.__len = int((X.groupby('id').size() - self.window_size).sum() / 
                        self.batch_size)
        del X


    def __len__(self):
        """Denotes the number of batches per epoch
        :return: number of batches per epoch
        """
        return int(self.__len / self.epoch_len_reducer)
    
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
            cut = random.randint(0, nrows - self.window_size)
            s = unit[cut: cut + self.window_size].T
            y =self._Y[sid][cut + self.window_size-1]
            _X.append(s)
            _y.append(y)

        
        _X = np.array(_X)
        if self.add_extra_channel:
            _X = _X.reshape(_X.shape + (1,))
            
        if self.noise_level > 0:
            noise_level = self.noise_level
            noise = np.random.normal(-noise_level, noise_level, _X.shape)
            _X = _X + noise
            _X = (_X - _X.min()) / (_X.max() - _X.min())
       
        if self.return_label:
            return _X, np.array(_y).reshape((self.batch_size, 1))
        elif self.output_extra_channel:
            return _X, _X.reshape(_X.shape + (1,))
        else:
            return _X, _X
        
    def on_epoch_end(self):
        """Updates indexes after each epoch
        """
        pass

class Att2SignalGenerator(tf.keras.utils.Sequence):

    def __init__(self, attributes, batches_per_epoch=1000, batch_size=32, 
                 expand_dims=False, return_attributes=False, return_signal=True, 
                 return_rul=True):
        self.attributes = attributes
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch
        self.expand_dims = expand_dims
        self.return_attributes = return_attributes
        self.return_signal = return_signal
        self.return_rul = return_rul
        
        self._numattributes = attributes[0]['norm_attributes'].shape[0]
        self._numfeatures = max([a['feature'] for a in attributes]) + 1
        self._signal_length = attributes[0]['signal'].shape[0]
        
        self._indexes = list(range(0, len(attributes), self._numfeatures))
        np.random.shuffle(self._indexes)
            
    def __len__(self):
        return self.batches_per_epoch

    def __getitem__(self, idx):
        
        indexes = self._indexes[idx * self.batch_size:(idx+1) * self.batch_size]
        
        Y = np.zeros((self.batch_size,))
        X = np.zeros((self.batch_size, self._numfeatures, self._signal_length))
        A = np.zeros((self.batch_size, self._numfeatures, self._numattributes))
        for k, i in enumerate(indexes):
            X[k] = np.array([a['signal'] for a in self.attributes[i:i+self._numfeatures]])
            Y[k] = self.attributes[i]['y']
            
            for ai,j in enumerate(range(i, i+self._numfeatures)):
                A[k,ai] = self.attributes[j]['norm_attributes']

        if self.expand_dims:
            X = tf.expand_dims(X, axis=-1)
            
        
        R = []
        if self.return_signal:
            R.append(X)

        if self.return_attributes:
            R.append(tf.cast(A, dtype=tf.float32))
            
        if self.return_rul:
            R.append(Y)
            

            
        return tuple(R)
    
    def on_epoch_end(self):
        np.random.shuffle(self._indexes)
            
    
# todo: no tomar el punto medio, darle aletoriedad respecto al punto medio (siguiento una gausiana)
def batch_discretization(values, bins):
    """
    Takes a set of values and divides them into equal-sized bins, replacing each 
    value with the midpoint value of the corresponding bin. The output is the 
    discretized values.
    
    
    """

    v_tf = tf.cast(values, dtype=tf.float32)

    v_max, v_min = tf.reduce_max(v_tf), tf.reduce_min(v_tf)
    step = (v_max - v_min) / bins

    # Create the range values
    min_values = tf.cast(tf.range(v_min, v_max + step/2, step), dtype=tf.float32)
    max_values = tf.cast(min_values + step, dtype=tf.float32)

    # Expand dimensions for broadcasting
    min_values = tf.expand_dims(min_values, axis=1)
    max_values = tf.expand_dims(max_values, axis=1)
    v_tf = tf.expand_dims(v_tf, axis=0)

    # Create masks for elements within the ranges
    masks = tf.logical_and(tf.greater_equal(v_tf, min_values), 
                           tf.less(v_tf, max_values))

    # Compute the value to replace within each range
    replace_values = min_values + ((max_values - min_values) / 2)

    # Return de values replaced
    return tf.squeeze(tf.gather(replace_values, tf.where(tf.transpose(masks))[:, 1]))
