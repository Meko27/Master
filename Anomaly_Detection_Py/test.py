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

print('test successfull')
path_normal = 'C:\\Users\\welschm_2\\Desktop\\MK\\Masterarbeit\\data\\carpet\\train\\good\\000.png'
path_normal = '/mnt/c/Users/welschm_2/Desktop/MK/Masterarbeit/data/carpet/train/good/000.png'
path normal = '/home/Meko/Repos/Master/Anomaly_Detection_Py/data/carpet/train/good/000.png'
img = cv2.imread(path_normal)
print(img.shape)
plt.imshow(img)
plt.show()
