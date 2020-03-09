# Carpet images anomaly detection
import numpy as np
import cv2
import os
import glob
import ntpath
from skimage.feature import hog
import skimage
import matplotlib.pyplot as plt
import time
import pandas as pd
import sys
sys.path.append(os.getcwd() + '/Source') # add source path to pythonpath
from AnoDetector import AnoDetector
from HOG_ground_dist import HOG_ground_dist



# Set path
###MacOS###
#path_normal = '/Users/meko/Documents/MATLAB/Anomaly_Detection/data/carpet/train/good'
#path_anomal = '/Users/meko/Documents/MATLAB/Anomaly_Detection/data/carpet/test/hole_cropped'
#path_normal = '/Users/meko/Documents/MATLAB/Anomaly_Detection/data/metal_nut/train/good'
#path_anomal = '/Users/meko/Documents/MATLAB/Anomaly_Detection/data/metal_nut/test/scratch'
###Windows###
#path_normal = 'C:\\Users\\welschm_2\\Desktop\\MK\\Masterarbeit\\data\\carpet\\train\\good'
#path_anomal = 'C:\\Users\\welschm_2\\Desktop\\MK\\Masterarbeit\\data\\carpet\\test\\hole'
###Windows Subsystem for Linux###
#path_normal = '/mnt/c/Users/welschm_2/Desktop/MK/Masterarbeit/data/carpet/train/good'
#path_anomal = '/mnt/c/Users/welschm_2/Desktop/MK/Masterarbeit/data/carpet/test/hole'
###VM###
#path_normal = '/home/mk/Master/Anomaly_Detection_Py/data/carpet/train/good'
#path_anomal = '/home/mk/Master/Anomaly_Detection_Py/data/carpet/test/hole'
###24core###
path_normal = '/home/Meko/Repos/Master/Anomaly_Detection_Py/data/carpet/train/good'
path_anomal = '/home/Meko/Repos/Master/Anomaly_Detection_Py/data/carpet/test/hole'
valid_img_type = '.png' # Set type
img_addrs_list_normal = glob.glob(path_normal + '/*' + valid_img_type) #Windows: '\\*' , Mac, WSL: '/*
img_addrs_list_anomal = glob.glob(path_anomal + '/*' + valid_img_type) #Windows: '\\* , Mac, WSL: '/*
img_addrs_list = img_addrs_list_normal + img_addrs_list_anomal
#img_addrs_list = img_addrs_list[:20] # TODO only for testing
img_id_list = []
img_list = []
hog_list = []


# Params
# Image scale
img_scale = 0.5
# HOG:
orientations = 9
pixels_per_cell = (128, 128)
cells_per_block = (2, 2)
block_stride = tuple(int(cell_size -1) for cell_size in cells_per_block)
# Ano detector:
iter = 3
n_outliers = 20
k = 3

# Load images
for i,addr in enumerate(img_addrs_list):
    img_id = ntpath.basename(addr)
    img_id = img_id[:-len(valid_img_type)]
    img_id_list.append(img_id) # list of img names
    img = cv2.imread(addr)
    #img = cv2.resize(img,tuple(int(img_scale * size) for size in img_size))
    #if i<len(img_addrs_list_normal):
    #    img = img[400:656,400:656]
    img = cv2.resize(img,(512,512))
    img_size = img.shape[:2]
    img_list.append(img)
    print('load img {} of {}'.format(i+1,len(img_addrs_list)))


# Calculate ground distance
print('img loaded')
print('calculate ground distance')
ground_dist,d_hog = HOG_ground_dist(img_list[0],cell_size=pixels_per_cell, block_size=cells_per_block,block_stride=block_stride,move_cost=1,rotation_cost=10,threshold=5)
print('ground_dist calculated, size:', ground_dist.shape)

# Extract Features
hog_mat = np.empty((len(img_addrs_list), int(d_hog)))
for i,img in enumerate(img_list):
    hog_vect = hog(img,orientations = orientations, pixels_per_cell = pixels_per_cell, cells_per_block = cells_per_block)
    hog_list.append(hog_vect)
    #hog_mat[i] = hog_vect

hog_mat = np.array(hog_list)
print('HOG calculated, HOG size: ', hog_mat.shape)

# Calculate Anomalies
x = hog_mat
Ano_Detector = AnoDetector(x,k=k,iter = iter,metric='emd',grounddist=ground_dist)
print('AnoDetector constructed')

start = time.time()
dist_vect_ano, idx_outliers,Ano_Detector = Ano_Detector.calc_outliers(n_outliers)
end = time.time()
time.sleep(5)

print('img_size: {}, HOG_size: {}'.format(img_size,hog_mat.shape))
print('dist: \n', dist_vect_ano)
print('idx_outl: \n', idx_outliers)
print('time elapsed: ', end-start)


ids_outliers = []
for idx in idx_outliers:
    ids_outliers.append(img_id_list[idx])

Profil1_evaluation = {'Id': ids_outliers, 'Distance': dist_vect_ano}
Profil1_evaluation = pd.DataFrame(data=Profil1_evaluation)
print('Eval ', Profil1_evaluation)

#for i in range(n_outliers):
#    fig,ax = plt.subplot(4,5,i+1)
#    idx = idx_outliers[i]
#    plt.imshow(img_list[idx])
#    plt.title('img: {}'.format(img_id_list[idx]))
#    ax.set_xticklabels = []
#    ax.set_ytocklabels = []

#plt.show()
