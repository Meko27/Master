# Metal Nut Anomaly detection

import numpy as np
import cv2
import os
import glob
import ntpath
from skimage.feature import hog
from skimage import exposure, img_as_float, img_as_ubyte
import matplotlib.pyplot as plt
import time
import pandas as pd
import sys
sys.path.append(os.getcwd() + '/Source') # add source path to pythonpath
from AnoDetector import AnoDetector
from HOG_ground_dist import HOG_ground_dist



def strech_contrast(img,A=0.15,B=0.65,SA=0.3,SB=0.9):
    # Stech Contrast of Grayscale Image

    img_float = img_as_float(img)  # Scale into range [0, 1] of float

    mask_left = (img_float < A)
    mask_center = ((img_float >= A) & (img_float < B))  
    mask_right = (img_float >= B)

    img_cont = np.copy(img_float)          
    img_cont[mask_left]   = img_float[mask_left] * SA / A
    img_cont[mask_center] = SA + (img_float[mask_center]-A) * (SB - SA) / (B - A)
    img_cont[mask_right] = SB + (img_float[mask_right]-B) * (1. - SB) / (1. - B)
    img_cont = img_as_ubyte(img_cont)
    
    return img_cont


###########################################################
### Set Parameters
# Preprocessing
RESIZE_FACTOR = 0.7
# HOG
orientations = 9
pixels_per_cell = (128,128)
cells_per_block = (2, 2)
block_stride = tuple(int(cell_size -1) for cell_size in cells_per_block)
# Ano detector:
iter = 5
n_outliers = 20
k = 7
###########################################################

###MacOS###/
#path_data = '/Users/meko/Downloads/Technoform_profiles/'
###24core###
path_data = '/home/Meko/Repos/data/bottle/bottle'

# Load images
valid_img_type = 'png'
img_addrs_list = glob.glob(path_data + '/train/good/' + '*' + valid_img_type)
img_addrs_list = img_addrs_list + glob.glob(path_data + '/test/broken_large/' + '*' + valid_img_type)

#img_addrs_list = img_addrs_list[:20] # TODO only for testing
img_ids = []
img_list = []
edges = []
width_sum = 0
# Load image and do preprocessing
CLIP_LIMIT = 0.015 # between (0,1) higher value -> higher contrast

# Load images and detect edges
for i,addr in enumerate(img_addrs_list):
    
    # Load image
    img_id = ntpath.basename(addr)
    img_ids.append(img_id[:-len(valid_img_type)])
    img = cv2.imread(addr)

    # Contrast Limited Adaptive Histogram Equalizaton 
    #img = exposure.equalize_adapthist(img,clip_limit = CLIP_LIMIT) 
    #img = img_as_ubyte(img) # equalize_adapthist converst image to float64

    # Resize
    img_re = cv2.resize(img,(int(img.shape[1]*RESIZE_FACTOR) , int(img.shape[0]*RESIZE_FACTOR))) 
    img_list.append(img_re)
    print('load img {} of {}'.format(i+1,len(img_addrs_list)))
    

# Calculate ground distance
print('Calculate Ground Distane Matrix')
ground_dist,d_hog = HOG_ground_dist(img_list[0],cell_size=pixels_per_cell,
                                    block_size=cells_per_block,
                                    block_stride=block_stride,
                                    move_cost=1,rotation_cost=5,threshold=7)
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

print('img_size: {}, HOG_size: {}'.format(img_list[0].shape,features.shape))
print('dist: \n', dist_vect_ano)
print('idx_outl: \n', idx_outliers)
print('time elapsed: ', end-start)
