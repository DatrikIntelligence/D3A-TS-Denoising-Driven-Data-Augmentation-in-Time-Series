import tqdm
import numpy as np
from d3a.meta import ATTRIBUTE_NAMES
import tensorflow as tf
import random 

NORM_ATTRIBUTE_NAMES = ['stability', 'periodicity', 'peculiarity', 'oscilatlion', 
                        'complexity', 'simetry', 'slope',  'informative', 'peaks', 
                        'noise', 'dynamic_range', 'standard_deviation', 'variability']

def norm(att , ifeature, min_max, att_names):
    r = np.zeros((len(att_names),))
    for i, attname in enumerate(att_names):
        d = min_max[f"{attname}_{ifeature}"]
        if len(d) == 2:
            _min, _max, = min_max[f"{attname}_{ifeature}"]
        else:
            _min, _max, _, _ = min_max[f"{attname}_{ifeature}"]

        v = (att[i] - _min) / (_max - _min + 1e-7)
        r[i] = v
    
    return r

def get_SAT_arrays(attributes, min_max, tslen):
    """
    Return (S)ignal, (A)ttributes and (T)arget arrays.
    """
    nsplits = tslen // 32

    S = []
    A = np.zeros( (len(attributes), len(NORM_ATTRIBUTE_NAMES)*nsplits + 4) ) 
    R = []
    i = 0
    
    for data in tqdm.tqdm(attributes):
        s = data['signal']
        atts = []
        for att in data['attributes']: 
            atts_array = np.array([att[a] for a in NORM_ATTRIBUTE_NAMES])
            mm = np.array([att[a] for a in ['min_value', 'max_value']])

            n = norm(atts_array, data['feature'], min_max, NORM_ATTRIBUTE_NAMES)

            atts += list(n)

        #OMG! sample normalization
        #s = (s - s.min()) / (s.max() - s.min() + 1e-12)

        # add init and end point of the time series
        a = np.array(atts + [s.min(), s.max(), s[0], s[-1]])

        data['norm_attributes'] = a


        S.append(s)
        A[i] = a
        R.append(data['y'])

        i+= 1
        
        
    S = np.array(S)
    R = np.array(R)
    
    mask = (~np.any(np.isnan(S), axis=1)) & (~np.any(np.isnan(A), axis=1))  
    
    S = S[mask]
    R = R[mask]
    A = A[mask]
        
    return np.array(S),A,np.array(R)

class A2SDiffusionGenerator(tf.keras.utils.Sequence):
    
    def __init__(self, difussion_model, base_generator, denoise_steps, noise_level,raw_samples_factor):
        self.base_generator = base_generator
        self.difussion_model = difussion_model
        self.denoise_steps = denoise_steps
        self.noise_level = noise_level
        self.raw_samples_factor = raw_samples_factor
        
        assert denoise_steps >= 0

        
    def __len__(self):
        return len(self.base_generator)
    
    def __getitem__(self, index):
        

        S, A, Y = self.base_generator.__getitem__(index)
       
        X = np.zeros(shape=S.shape)
        for f in range(X.shape[1]):
            s = tf.cast(S[:,f,:], dtype=tf.float32)
            if isinstance(self.noise_level, float):
                noise_rates = self.noise_level
            else:
                noise_rates = self.difussion_model.noise_rates[self.noise_level] 
                
            if random.uniform(0, 1) > self.raw_samples_factor and noise_rates > 0:
                a = A[:, f]
                

                #print(noise_rates)
                signal_rates = 1 - noise_rates
                noise = np.random.normal(size=s.shape) * noise_rates
                noisy_samples = noise + s * signal_rates
                
                for i in range(self.denoise_steps, 0, -1):
                    noisy_samples, _, _ = self.difussion_model.neighbourhood(1, 
                                                             noisy_samples=noisy_samples, 
                                                             signal_rates=signal_rates, 
                                                             noise_rates=noise_rates,
                                                             timesteps=np.array([i-1]),
                                                             features=a)
            
                X[:, f, :] = noisy_samples
                
                #assert not np.any(np.isnan(x))
                
            else:
                X[:, f, :] = s

        
        return np.nan_to_num(X), Y


        
    def on_epoch_end(self):
        self.base_generator.on_epoch_end()

        
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
