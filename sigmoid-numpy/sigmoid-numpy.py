import numpy as np

def sigmoid(x):
    y=np.array(x)
    return (1/(1+np.exp(-y)))