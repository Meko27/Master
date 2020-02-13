'''
Module to calculate local distance matrix and weighted graph Laplacian

'''
import numpy as np
import cv2
from scipy.spatial.distance import pdist, squareform
from numpy.matlib import repmat
import time
from pdist_emd import pdist_split
import ray
import itertools


################################################################################
#Auxiliary Functions
################################################################################
def chunked_iterable(iterable, size):
# Auxiliary function to iterate over chunks
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, size))
        if not chunk:
            break
        yield chunk

@ray.remote
def calc_emd(x1,x2,grounddist):
    emd,_,_ = cv2.EMD(x1,x2,cv2.DIST_USER,cost=grounddist)
    return emd


################################################################################

def local_dist_mat(x,k,metric='euclidean',grounddist=0,iteration=0):
# local_dist_mat: Calculates the knn - local distance matrix based on parameter k
    BATCH_SIZE = 500
    n_samples,n_dim = x.shape
    ixx = np.linspace(0,n_samples-1,n_samples,dtype=int)
    ixx = repmat(ixx,k,1)
    
    # check which metric to be calculated
    if metric == 'emd':
        x1,x2 = pdist_split(x)
        grounddist = np.float32(grounddist)
        x1 = np.float32(x1)
        x1_t = np.transpose(x1)
        x2 = np.float32(x2)
        x2_t = np.transpose(x2)
        n_pdist = len(x1)
        #emd_vect = np.empty(n_pdist)
        dist_vect = []
        emd = []
        #for i in range(n_pdist):
        #    emd_vect.append(calc_emd.remote(x1_t[:,i],x2_t[:,i],grounddist))
        #    print('Now calculating {} of {} at iteration {}'.format(i+1,n_pdist,iteration+1))
        #dist_vect = ray.get(emd_vect)

        for no_batch,batch in enumerate(chunked_iterable(range(n_pdist),size=BATCH_SIZE)):
            start = time.time()
            for i in batch:
                emd.append(calc_emd.remote(x1_t[:,i],x2_t[:,i],grounddist)) # calculate EMD's of batch
            dist_vect = ray.get(emd)
            end = time.time()
            print('Now calculating batch {} of {} at iteretion {}'.format(no_batch+1 , n_pdist//BATCH_SIZE, iteration+1))
            print('Calculation time per batch: {} (Batch size: {})'.format(end-start , BATCH_SIZE))

    else:
        dist_vect = pdist(x,metric)
    
    dist_mat = squareform(dist_vect) 
    
    # calculate knn dist matrix 
    knn_mat = np.argsort(dist_mat,axis=0) 
    knn_mat = knn_mat[1:k+1,:] # only first k elements (without the zeros)
    knn_dist_mat = np.zeros((k,n_samples))
    for j in range(n_samples):
        for i in range(k):
            idx_row = knn_mat[i,j]
            idx_col = j
            knn_dist_mat[i,j] = dist_mat[idx_row,idx_col]

    dist_mat_local = np.zeros((n_samples,n_samples))
    for j in range(n_samples):
        for i in range(k):
            idx_row = knn_mat[i,j]
            idx_col = ixx[i,j]
            dist_mat_local[idx_row,idx_col] = knn_dist_mat[i,j]

    dist_mat_local = np.maximum(dist_mat_local,np.transpose(dist_mat_local)) # symmetrize

    return dist_mat_local



def weighted_graph_laplacian(dist_mat):
# local_graph_laplacian: Calculates the weighted graph laplacian of a distance matrix
    n = dist_mat.shape[0]
    d = np.sum(dist_mat,axis=1) # sum up rows (-->)
    if len(d[d==0]) > 0:
        i_zeros = np.where(d==0)
        d[i_zeros] = 1
    D = np.diag(1/d) # Degree Matrix
    A_t1 = np.matmul(D,dist_mat)
    A_t2 = np.matmul(A_t1,D)
    print('A_t1: {}, A_t2: {}'.format(np.max(A_t1),np.max(A_t2)))
    A = 1.0/n * np.matmul(np.matmul(D,dist_mat),D) # weighted Adjacency matrix
    print('max_A: ', np.max(A))
    d = np.sum(A,axis=1) # sum up rows (-->)
    print('max_d', np.max(d))
    if len(d[d==0]) > 0:
        i_zeros = np.where(d==0)
        d[i_zeros] = 1/n
    print('D: {}, d: {}'.format(np.max(D),np.max(d)))    
    D = np.diag(d) # Degree Matrix of weighted A
    
    L = D-A # Graph Laplacian

    return L
