# Technoform Profiels Anomaly Detection

import numpy as np
import cv2
import os
import glob
import ntpath
from skimage.feature import hog
import skimage
import matplotlib.pyplot as plt
from AnoDetector import AnoDetector
from HOG_ground_dist import HOG_ground_dist
import time
import pandas as pd

###########################################################
### Set Parameters
# Preprocessing
RESIZE_FACTOR = 0.4
# HOG
orientations = 9
pixels_per_cell = (64,64)
cells_per_block = (2, 2)
block_stride = tuple(int(cell_size -1) for cell_size in cells_per_block)
# Ano detector:
iter = 10
n_outliers = 10
k = 7
###########################################################

###MacOS###
#path_data = '/Users/meko/Documents/Repos/Master/Anomaly_Detection_Py/Technoform_profiles/data/Profile1'
###24core###
path_data = '/home/Meko/Repos/Master/Anomaly_Detection_Py/Technoform_profiles/data/Profil_2/'

# Load images
valid_img_type = 'bmp'
img_addrs_list = glob.glob(path_data + '/*' + valid_img_type)
#img_addrs_list = img_addrs_list[:20] # TODO only for testing
img_ids = []
img_list = []
for i,addr in enumerate(img_addrs_list):
    img_id = ntpath.basename(addr)
    img_ids.append(img_id[(len(img_id)-6):-len(valid_img_type)])
    img = cv2.imread(addr)
    img = img[:,750:2000]
    img_size = img.shape[:2]
    img = cv2.resize(img,tuple(int(size * RESIZE_FACTOR) for size in img_size))
    img_list.append(img)
    print('load img {} of {}'.format(i+1,len(img_addrs_list)))

# Calculate ground distance
print('Calculate Ground Distane Matrix')
ground_dist,d_hog = HOG_ground_dist(img_list[0],cell_size=pixels_per_cell, 
                                    block_size=cells_per_block,
                                    block_stride=block_stride)
print('Ground Distance Matrix calculated, size:', ground_dist.shape)

# Extraxt Features
features = []
for i,img in enumerate(img_list):
    hog_vect = hog(img,orientations = orientations, pixels_per_cell = pixels_per_cell, cells_per_block = cells_per_block)
    features.append(hog_vect)

features = np.array(features)
print('HOGs calculated, feature mat size: ', features.shape)

# Calculate Anomalies
x = features
print('Construct Ano Detector Object')
Ano_Detector = AnoDetector(x,k=k,iter = iter,metric='emd',grounddist=ground_dist)
print('AnoDetector constructed')

start = time.time()
dist_vect_ano, idx_outliers,Ano_Detector = Ano_Detector.calc_outliers(n_outliers)
end = time.time()
time.sleep(5)

ids_outliers = []
for idx in idx_outliers:
    ids_outliers.append(img_ids[idx])

Profil1_evaluation = {'Id': ids_outliers, 'Distance': dist_vect_ano}
Profil1_evaluation = pd.DataFrame(data=Profil1_evaluation)
print('Eval ', Profil1_evaluation)

print('img_size: {}, HOG_size: {}'.format(IMG_SIZE,features.shape))
print('dist: \n', dist_vect_ano)
print('idx_outl: \n', idx_outliers)
print('time elapsed: ', end-start)
