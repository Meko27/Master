# Carpet images anomaly detection
import numpy as np
import cv2
import os
import glob
import ntpath
import skimage
import matplotlib.pyplot as plt
from AnoDetector_wrapper import AnoDetector
import time
from crop import crop
import metric


# Set path
###MacOS###
#path_normal = '/Users/meko/Documents/MATLAB/Anomaly_Detection/data/carpet/train/good'
#path_anomal = '/Users/meko/Documents/MATLAB/Anomaly_Detection/data/carpet/test/hole'
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

valid_img_type = '.png' # all images are type jpg
img_addrs_list_normal = glob.glob(path_normal + '/*' + valid_img_type) #Windows: '\\*' , Mac, WSL: '/*
img_addrs_list_anomal = glob.glob(path_anomal + '/*' + valid_img_type) #Windows: '\\* , Mac, WSL: '/*
img_addrs_list = img_addrs_list_normal + img_addrs_list_anomal
img_addrs_list = img_addrs_list[:20] # TODO only for testing
img_id_list = []
img_list = []
hog_list = []


# Params
# Cropping images
SLICENUMBER = 2
# Image scale
img_scale = 1
# HOG:
orientations = 9
pixels_per_cell = (64, 64)
cells_per_block = (2, 2)
block_stride = tuple(int(cell_size -1) for cell_size in cells_per_block)
# Ano detector:
iter = 10
n_outliers = 17
k = 7

# Load images
for i,addr in enumerate(img_addrs_list):
    img_id = ntpath.basename(addr)
    img_id = img_id[:-len(valid_img_type)]
    img_id_list.append(img_id) # list of img names
    img = cv2.imread(addr)
    img_size = img.shape[:2]
    img = cv2.resize(img,tuple(int(img_scale * size) for size in img_size))
    #img = cv2.resize(img,(512,512))
    img_size = img.shape[:2]
    img_list.append(img)
    print('load img {} of {}'.format(i+1,len(img_addrs_list)))


img_cropped_list = []
# Crop images
for img in img_list:
    try:
        img_cropped = crop(img,SLICENUMBER)
        img_cropped_list.append(img_cropped)
    except:
        print('Cropfactor does not fit image size')



# Calculate ground distance
print('img loaded')
print('calculate ground distance')
try:
    ground_dist = metric.EMD_details.ground_distance_matrix_of_2dgrid(img_size[1], img_size[0])
    print('ground_dist calculated, size:', ground_dist.shape)
except:
    print('Grounddistance could not be calculated')

# Extract Features

img_reshaped_len = img_size[0]*img_size[1]
img_reshaped_list = []
for img_slices in img_cropped_list:
    for img in img_slices:
        img_reshaped = np.reshape(img, img_reshaped_len, order='F')
        img_reshaped_list.append(img_reshaped)
    

img_mat = np.array(img_reshaped_list,dtype=np.int8)
print('img-mat size: ', img_reshaped.shape)

# Calculate Anomalies
x = img_reshaped
Ano_Detector = AnoDetector(x,k=k,iter = iter,metric='emd',grounddist=ground_dist)
print('AnoDetector constructed')

start = time.time()
dist_vect_ano, idx_outliers,Ano_Detector = Ano_Detector.calc_outliers(n_outliers)
end = time.time()
time.sleep(5)

print('img_size: {}, HOG_size: {}'.format(img_size,hog_mat.shape))
print('dist: \n', dist_vect_ano)
print('idx_outl: \n', np.ceil(idx_outliers/SLICENUMBER))
print('time elapsed: ', end-start)

for i in range(n_outliers):
    plt.subplot(4,5,i+1)
    idx = idx_outliers[i]
    plt.imshow(img_cropped_list[idx])
    plt.title('img: {}'.format(np.ceil(img_id_list[idx]/SLICENUMBER)))

plt.show()
