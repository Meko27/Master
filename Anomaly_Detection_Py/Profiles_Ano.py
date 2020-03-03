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
#import pandas as pd

###########################################################
### Set Parameters
# Preprocessing
RESIZE_FACTOR = 1
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
path_data = '/Users/meko/Downloads/Technoform_profiles/Profil_1'
###24core###
#path_data = '/home/Meko/Repos/Master/Anomaly_Detection_Py/Technoform_profiles/data/Profil_2/'

# Load images
valid_img_type = 'bmp'
img_addrs_list = glob.glob(path_data + '/*' + valid_img_type)
#img_addrs_list = img_addrs_list[:20] # TODO only for testing
img_ids = []
img_list = []
# Load image and do preprocessing
CROP_BOUNDX = 200 # precroppung to delete bounds
CROP_BOUNDY = 300
FILTER_LEN = 100 # Object detection
FILTER_WID = 5
for i,addr in enumerate(img_addrs_list):
    start = time.time()
    img_id = ntpath.basename(addr)
    img_ids.append(img_id[(len(img_id)-6):-len(valid_img_type)])
    img = cv2.imread(addr)

    # preprocessing 
    img_g = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) # to gray
    img_g_crop = img_g[CROP_BOUNDY:-CROP_BOUNDY , CROP_BOUNDX:-CROP_BOUNDX]
        # contrast streching
    ALPHA = 2
    BETA = 20
    img_cont = np.zeros((img_g_crop.shape[0],img_g_crop.shape[1]),dtype = 'uint8')
    for i in range(img_g_crop.shape[0]):
        for j in range(img_g_crop.shape[1]):
            img_cont[i,j] = np.clip(ALPHA*img_g_crop[i,j]+BETA,0,255)
    # Detect Object
    kernel = np.ones((FILTER_LEN,FILTER_WID), dtype=np.uint8)
    img_centery = img_cont.shape[0]//2 # vertical img center
    pixel_pos = []
    for i in range(FILTER_WID//2 + 1 , img_cont.shape[1] - FILTER_WID//2):
        result = kernel * img_cont[img_centery - FILTER_LEN//2 : img_centery + FILTER_LEN//2 , i-FILTER_WID//2 : i+FILTER_WID//2 + 1 ]
        if np.sum(result) < FILTER_WID*FILTER_LEN*200:
            pixel_pos.append(i)
    print('min,max', (pixel_pos[0],pixel_pos[-1]))        
    bound_left = pixel_pos[0] + CROP_BOUNDX - 10
    bound_right = pixel_pos[-1] + CROP_BOUNDX + 10
    img_out = img[:,bound_left:bound_right]
    
    # resize
    img_width = img_out.shape[1]
    img_heigth = img_out.shape[0]
    img_re = cv2.resize(img_out,(img_width*RESIZE_FACTOR , img_heigth*RESIZE_FACTOR))
    img_list.append(img_re)
    print('load img {} of {}'.format(i+1,len(img_addrs_list)))
    print('left bound: {}, right bound: {}'.format(bound_left,bound_right))
    end=time.time()
    print('time: ',end-start)
    plt.subplot(1,4,1)
    plt.imshow(img)
    plt.title('original')
    plt.subplot(1,4,2)
    plt.imshow(img_cont,cmap='gray')
    plt.title('contrasted imag')
    plt.subplot(1,4,3)
    plt.imshow(img_re)
    plt.title('final cropped')
    plt.subplot(1,4,4)
    plt.imshow(img_out)
    plt.title('out')
    plt.show()

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
