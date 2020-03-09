import numpy as np
import cv2
import timeit
import ray


def pdist_split(x):

    n_samples,dim = x.shape
    n_pdist = int(n_samples*(n_samples-1)/2)
    x1 = np.empty((n_pdist,dim))
    x2 = np.empty((n_pdist,dim))
    idx = 0
    for j in range(n_samples-1):
        start = j+1
        for i in range(start,n_samples):
            x1[idx] = x[i,:]
            x2[idx] = x[j,:]
            idx += 1
    return x1,x2

