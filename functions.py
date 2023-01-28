import numpy as np
from tslearn.metrics import dtw
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from sklearn.metrics.pairwise import manhattan_distances
from scipy.signal import correlate
from scipy.interpolate import interp1d

def repeat_array(array, times):
        arr = list(array)
        for m in range(times):
            for i in range(len(array)):
                arr.append(arr[i])
        return np.array(arr)

def cross_correlation(arr1, arr2):
    return max(correlate(arr1, arr2))

def euclidean_distance(arr1, arr2):
    return np.sqrt(sum((np.array(arr1)-np.array(arr2))**2))

def fdtw(arr1, arr2):
    distance, path = fastdtw(arr1, arr2)
    return distance

def interpol(array, length):
        return interp1d(np.arange(0, len(array)), array)(np.linspace(0.0, len(array)-1, length))
    
def mean(array):
    return sum(array)/len(array)

def integer_mean(array):
    return sum(array)//len(array)

def center_moving_average(array, period = 3):
    ret = np.cumsum(array, dtype=float)
    ret[period:] = ret[period:] - ret[:-period]
    return ret[period - 1:] / period

def pad(array, length):
    return np.pad(array, (0,length - len(array)), 'constant')

functions = {'euclidean' : euclidean,
             'dtw' : dtw,
             'fdtw' : fdtw,
             'correlation' : cross_correlation,
             'interpolate' : interpol,
             'cma' : center_moving_average,
             'avg' : mean,
             'int_avg' : integer_mean,
             'pad' : pad,
             'reinterpolate' : repeat_array
             }