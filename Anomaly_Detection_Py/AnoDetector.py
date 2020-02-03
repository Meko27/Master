import numpy as np

import cv2
import timeit
from Diffusion_Fct import local_dist_mat, weighted_graph_laplacian
import ray

class AnoDetector:

    def __init__(self, raw_data, 
                k=10, 
                iter=10, 
                metric='euclidean', 
                grounddist=0):
        self.k = k
        self.iter = iter
        self.metric = metric
        self.grounddist = grounddist
        self.raw_data = raw_data
        self.data_encoded = []

    def calc_outliers(self,n_outliers):
        n = self.raw_data.shape[0] 
        # Denoising Process
        L_vect = []
        ray.init() # initialize ray for parallel processing
        for i in range(self.iter):
            print('Iter: ', i+1)
            local_dist = local_dist_mat(self.raw_data,
                                        self.k,
                                        self.metric,
                                        self.grounddist,
                                        iteration=i)                              
            L = weighted_graph_laplacian(local_dist)
            L_vect.append(L)
            D = np.diag(np.diag(L_vect[i]))
            A = D+0.25*L_vect[i]
            B = np.matmul(D,self.raw_data)
            print('shape A: {}, shape B: {}'.format(A.shape,B.shape))
            self.raw_data = np.linalg.solve(A,B)
            self.data_encoded.append(self.raw_data)

        # Calculate distances of everey iteration
        d = []
        for i in range(self.iter):
            if i==0:
                d.append(np.absolute(self.data_encoded[i] - self.raw_data))
                d_total = d[i]
            else:
                d.append(np.absolute(self.data_encoded[i] - self.data_encoded[i-1]))
                d_total += d[i]
        
        d_total_eucl = np.sqrt(np.sum(d_total ** 2, axis=1))

        # Calculate largest distance
        self.idx_outliers = np.argsort(d_total_eucl)[-n_outliers:]
        self.idx_outliers = self.idx_outliers[::-1]
        self.dist_vect_ano = np.sort(d_total_eucl)[-n_outliers:]
        self.dist_vect_ano = self.dist_vect_ano[::-1]
        self.idx_norm = np.argsort(d_total_eucl)[:n_outliers]
        self.dist_vect_norm = np.sort(d_total_eucl)[:n_outliers]
        dist_vect_ano = self.dist_vect_ano
        idx_outliers = self.idx_outliers

        return dist_vect_ano, idx_outliers, self 

    def get_idx_norm(self):
        idx_norm = self.idx_norm
        return idx_norm

    def get_dist_vect_norm(self):
        dist_vect_norm = self.dist_vect_norm
        return dist_vect_norm       
            
