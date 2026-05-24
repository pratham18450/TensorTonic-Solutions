import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    weights=np.zeros(X.shape[1]);
    bias=0.0;
    for i in range(steps):
        Z=X@weights+bias
        A=_sigmoid(Z)
        delW=(X.T@(A-y))/X.shape[0]
        delb=np.sum(A-y)/X.shape[0]
        weights=weights-lr*delW
        bias=bias-lr*delb
    return (weights,bias)