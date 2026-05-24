import numpy as np

def matrix_transpose(A):
    A=np.array(A)
    ans=np.zeros((A.shape[1],A.shape[0]));
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            ans[j][i]=A[i][j]
    return ans