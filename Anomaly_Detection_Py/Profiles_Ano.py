# Technoform Profiels Anomaly Detection

import numpy as np
import cv2
import os
import glob
import ntpath
from skimage.feature import hog
from skimage import exposure, img_as_float, img_as_ubyte
import matplotlib.pyplot as plt
from AnoDetector import AnoDetector
from HOG_ground_dist import HOG_ground_dist
import time
#import pandas as pd


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

def contrast_bright_correction(img,ALPHA=2,BETA=20):
    # Contrast and Brightness corrections  
    
    img_cont = np.zeros((img.shape[0],img.shape[1]),dtype = 'uint8')
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img_cont[i,j] = np.clip(ALPHA*img[i,j]+BETA,0,255)  

    return img_cont

def detect_object(img , FILTER_LEN=800 , FILTER_WID=5):
    # Detect Objectbounds
    kernel = np.ones((FILTER_LEN,FILTER_WID), dtype=np.uint8)
    img_centery = img.shape[0]//2 # vertical img center
    pixel_pos = []
    for i in range(FILTER_WID//2 + 1 , img.shape[1] - FILTER_WID//2):
        result = kernel * img[img_centery - FILTER_LEN//2 : img_centery + FILTER_LEN//2 , i-FILTER_WID//2 : i+FILTER_WID//2 + 1 ]
        if np.sum(result) < FILTER_WID*FILTER_LEN*200:
            pixel_pos.append(i)            
    edge_left = pixel_pos[0] + CROP_BOUNDX - 10
    edge_right = pixel_pos[-1] + CROP_BOUNDX + 10
    
    return (edge_left,edge_right)


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
img_orig_list = []
img_list = []
edges = []
width_list = []
# Load image and do preprocessing
CROP_BOUNDX = 300 # precroppung to delete bounds
CROP_BOUNDY = 300
CLIP_LIMIT = 0.015 # between (0,1) higher value -> higher contrast

# Load images and detect edges
for img_count,addr in enumerate(img_addrs_list):
    start = time.time()
    # Load image
    img_id = ntpath.basename(addr)
    img_ids.append(img_id[(len(img_id)-6):-len(valid_img_type)])
    img = cv2.imread(addr)
    img_orig_list.append(img)

    # Preprocessing 
    img_g = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) # to gray
    img_g = img_g[CROP_BOUNDY:-CROP_BOUNDY , CROP_BOUNDX:-CROP_BOUNDX] # crop out edges
    img_g_cont = strech_contrast(img_g) # contrast streching

    # Detect Object
    edges.append(detect_object(img_g_cont)) 
    width_list.append(edges[img_count][1] - edges[img_count][0])
    end = time.time()
    print('time loop1: ', end-start)

width_max = np.max(np.array(width_list)) # max object width

# Process and Crop images acc. to calculated edges
for i,img in enumerate(img_orig_list):
    start = time.time()
    img_out = img[:,edges[i][0]:edges[i][0]+width_max,:]

    # Contrast Limited Adaptive Histogram Equalizaton 
    img_out = exposure.equalize_adapthist(img_out,clip_limit = CLIP_LIMIT) 
    img_out = img_as_ubyte(img_out) # equalize_adapthist converst image to float64

    # Resize
    img_width = img_out.shape[1]
    img_heigth = img_out.shape[0]
    img_re = cv2.resize(img_out,(img_width*RESIZE_FACTOR , img_heigth*RESIZE_FACTOR))
    img_list.append(img_re)
    print('load img {} of {}'.format(i+1,len(img_addrs_list)))
    cv2.imwrite(str(i) + '.bmp',img_re)
    end=time.time()
    print('time loop 2: ', end-start)

    
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
