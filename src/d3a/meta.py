from tsfresh.feature_extraction import feature_calculators as fc
import tsfresh
import numpy as np
from scipy import signal, stats
import tqdm
import tensorflow as tf

ATTRIBUTE_NAMES = ['stability', 'periodicity', 'peculiarity', 'oscilatlion', 'complexity', 'simetry', 'slope', 
                   'informative', 'peaks', 'noise', 'dynamic_range', 'min_value', 'max_value', 'standard_deviation', 
                   'variability']


def detect_outliers(data, m = 2.):
    """
    Detect outliers in the given data using the modified Z-score method.

    Args:
    - data (np.ndarray): Input data.
    - m (float): Number of standard deviations to consider as a threshold for outliers.

    Returns:
    - np.ndarray: Boolean array indicating whether each data point is an outlier.
    """    
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else np.zeros(len(d))
    return s>m

def remove_outliers(a):
    """
    Remove outliers from the given array by replacing them with the minimum and maximum non-outlier values.

    Args:
    - a (np.ndarray): Input array.

    Returns:
    - np.ndarray: Array with outliers replaced.
    """    
    outliers_mask = detect_outliers(a)
    ov = a[outliers_mask] 
    nov = a[~outliers_mask] 
    min_ov = outliers_mask & (a < nov.min())
    max_ov = outliers_mask & (a > nov.max())
    a[min_ov] = nov.min()
    a[max_ov] = nov.max()
    
    return a


def compute_metaattributes(gen, tslen):
    """
    Compute meta-attributes for each sample in a generator.

    Args:
    - gen: Data generator.
    - tslen (int): Length of the time series.

    Returns:
    - Tuple: List of samples and list of dictionaries containing meta-attributes for each sample.
    """    
    attributes = []
    samples = []
    isample = 0
    for batch in tqdm.tqdm(gen):

        for x, y in zip(*batch):

            for i in range(x.shape[0]):
                s = x[i]

                att = {'attributes': [get_attributes(s[i:i+32]) for i in range(0, tslen, 32)]}
                #att = np.stack(att)

                #assert not any([np.isnan(v) for k,v in att.items()])
                #assert len(att.values()) > 0

                att['y'] = y
                att['sample'] = isample
                att['feature'] = i
                att['signal'] = s
                attributes.append(att)
                print('.', end='')

            isample += 1
        
    return samples, attributes

def maxmin_attributes(attributes):
    """
    Compute the maximum, minimum, mean, and standard deviation of each attribute across all features.

    Args:
    - attributes (list): List of dictionaries containing meta-attributes for each sample.

    Returns:
    - dict: Dictionary containing maximum, minimum, mean, and standard deviation for each attribute.
    """    
    N_FEATURES = max([a['feature'] for a  in attributes]) + 1
    
    MAX_MIN = {}
    for att in ATTRIBUTE_NAMES:

        for ifeature in range(N_FEATURES):
            print(att, "series:", ifeature)
            data = np.array([a[att] 
                             for d in attributes
                             for a in d['attributes'] 
                             if d['feature'] == ifeature]) 
            _max, _min = data.min(), data.max()
            _mean, _std = data.mean(), data.std()

            MAX_MIN[f"{att}_{ifeature}"] = (_max, _min, _mean, _std)
            
    return MAX_MIN


def calculate_coefficient_of_variation(x):
    """
    Calculate the coefficient of variation for a given signal.

    Args:
    - x (numpy.ndarray): Input signal.

    Returns:
    - float: Coefficient of variation.
    """
    # Calculate the mean and standard deviation of the signal x
    mean, std = np.mean(x), np.std(x)

    # Calculate the coefficient of variation of the signal x
    return std / (mean + 1e-12)


def calculate_entropy(x):
    """
    Calculate the entropy of a given signal.

    Args:
    - x (numpy.ndarray): Input signal.

    Returns:
    - float: Entropy of the signal.
    """
    # Calculate the probability distribution of the values in the signal x
    prob_distribution, _ = np.histogram(x, density=True)

    # Avoid the probability distribution containing zero values
    prob_distribution += 1e-12

    # Calculate the entropy of the signal x
    entropy = stats.entropy(prob_distribution)

    return entropy

def noise_ratio(a):
    """
    Calculate the signal-to-noise ratio for a given signal.

    Args:
    - a (numpy.ndarray): Input signal.

    Returns:
    - float: Signal-to-noise ratio.
    """
    mean = a.mean()
    std = a.std()
    return float(np.where(std == 0, 0, mean / std))


def complexity(s):
    """
    Calculate the complexity of a given signal using the Fourier transform.

    Args:
    - s (numpy.ndarray): Input signal.

    Returns:
    - float: Complexity of the signal.
    """
    # Normalize the signal
    s = (s - s.min()) / (s.max() - s.min() + 1E-7)

    # Calculate the Fourier transform of the signal
    X = np.fft.fft(s)

    # Calculate the magnitude of the Fourier transform
    magnitude = np.abs(X)

    # Calculate the complexity of the signal as the sum of the magnitude of the Fourier transform
    # per unit of time
    complexity = np.sum(magnitude)

    return complexity

def calculate_oscillation(x):
    """
    Calculate the oscillation level of a given signal.

    Args:
    - x (numpy.ndarray): Input signal.

    Returns:
    - float: Oscillation level of the signal.
    """
    # Calculate the standard deviation of the signal
    std = np.std(x)

    # Calculate the arithmetic mean of the signal
    _mean = np.mean(x)

    # Calculate the oscillation level as the ratio between the standard deviation and the mean
    oscillation = std / (_mean + 1E-7)

    return np.abs(oscillation)



def calculate_stability(x):
    """
    Calculate the stability of a given signal using linear regression.

    Args:
    - x (numpy.ndarray): Input signal.

    Returns:
    - float: Stability of the signal.
    """
    # Calculate the linear regression of the signal x
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y=np.arange(0, len(x)))

    # Calculate the stability as the absolute value of the coefficient of determination
    stability = np.abs(r_value**2)

    return stability


def evaluate_periodicity(x, num_periods):
    """
    Evaluate the periodicity of a given signal.

    Args:
    - x (numpy.ndarray): Input signal.
    - num_periods (int): Number of periods to consider for periodicity evaluation.

    Returns:
    - float: Periodicity of the signal.
    """
    # Calculate the number of complete periods in the signal x
    period = len(x) // num_periods

    # Divide the signal x into as many complete periods as possible
    periods = np.split(x[:period*num_periods], num_periods)

    # Calculate the similarity between each pair of consecutive periods
    similarities = []
    for i in range(num_periods-1):
        l, r = periods[i], periods[i+1]
        if np.all(l == r):
            similarity = 1
        elif np.all(l == l[0]) and np.all(r == r[0]):
            similarity = 1
        elif np.all(l == l[0]) or np.all(r == r[0]):
            similarity = 0
        else:
            similarity = np.corrcoef(periods[i], periods[i+1])[0, 1]
        similarities.append(similarity)
    
    # Calculate the periodicity as the mean of the similarities between consecutive periods
    periodicity = np.nanmean(np.abs(similarities))

    return periodicity


def get_attributes(s):
    
    return {
        'periodicity': evaluate_periodicity(s, 4),
        'stability': calculate_stability(s),
        'oscilatlion': calculate_oscillation(s),
        'complexity': complexity(s),
        'noise': noise_ratio(s),
        'informative': calculate_entropy(s),
        'variability': calculate_coefficient_of_variation(s),
        'standard_deviation': fc.standard_deviation(s),
        'peculiarity': fc.kurtosis(s),
        'dynamic_range': abs(s.max() - s.min()),
        'simetry': abs(fc.skewness(s)),
        'peaks': fc.number_cwt_peaks(s, 10),
        'slope': fc.linear_trend(s, [{'attr': 'slope'}])[0][1],
        'max_value': s.max(),
        'min_value': s.min(),
    }



def create_S2A_model(tslen, nattributes, lr, activation, output):
    """
    Create a Signal-to-Attribute (S2A) model.

    Args:
    - tslen (int): Length of the input time series.
    - nattributes (int): Number of attributes to predict.
    - lr (float): Learning rate for the optimizer.
    - activation (str): Activation function for hidden layers.
    - output (str): Activation function for the output layer.

    Returns:
    - tf.keras.models.Sequential: S2A model.
    """
    
    signal2attribute = tf.keras.models.Sequential()
    signal2attribute.add(tf.keras.Input(shape=(tslen,)))
    signal2attribute.add(tf.keras.layers.Dense(32, activation=activation))
    signal2attribute.add(tf.keras.layers.Dense(64, activation=activation))
    signal2attribute.add(tf.keras.layers.Dense(128, activation=activation))
    signal2attribute.add(tf.keras.layers.Dense(nattributes, activation=output))

    loss = tf.keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error") 
    opt = tf.keras.optimizers.Adam(lr=lr)
    signal2attribute.compile(optimizer=opt, loss=loss, 
                             metrics=[tf.keras.metrics.MeanAbsoluteError(name='mae')],
                 )
    
    return signal2attribute
