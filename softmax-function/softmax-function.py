import numpy as np

def softmax(x):
    ex=np.exp(x-np.max(x,axis=-1,keepdims=True));
    return ex/np.sum(ex,axis=-1,keepdims=True)