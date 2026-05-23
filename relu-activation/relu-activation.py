import numpy as np

def relu(x):
    y=np.array(x);
    return np.maximum(0,y)