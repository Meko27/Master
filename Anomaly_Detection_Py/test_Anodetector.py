#
from AnoDetector import AnoDetector
import numpy as np
import cv2


x = np.array([[2,8,8],[9,3,5],[9,7,1],[5,2,4],[4,2,1],[9,2,9],[7,4,9],[6,7,5],[8,7,1],[8,7,6]],dtype=np.float32)
x = x[:5]
k = 3
iter = 10
n_outliers = 2
ground_dist = np.array([[0,1,2],[1,0,1],[2,1,0]],dtype=np.float32)
#x_1 = np.transpose(x[1,:])
#x_1 = np.float32(x_1)
#x_2 = np.transpose(x[2,:])
#x_2 = np.float32(x_2)

#dist_mat = AnoDetector.local_dist_mat(x,k,metric='emd',grounddist=ground_dist)

Ano_Detector = AnoDetector(x,k=k,iter = iter,metric='emd',grounddist=ground_dist)
dist_vect_ano, idx_outliers,Ano_Detector = Ano_Detector.calc_outliers(n_outliers)
print('dist: \n', dist_vect_ano)
print('idx_outl: \n', idx_outliers)
#print('dist_mat= \n', dist_mat)
#L = AnoDetector.weighted_graph_laplacian(dist_mat)
#print(' L = \n', L)