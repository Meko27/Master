# Image Preprocessing

import cv2 
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
from skimage.filters import threshold_otsu
from scipy import ndimage

"""Applying CLAHE to resolve uneven illumination"""


def Clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    cl1 = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return cl1


# read image
path = '/Users/meko/Downloads/Technoform_profiles/Profil_1/'
im_id = 'Basler_acA3088-57uc__23186829__20200218_120749566_1.bmp'
img_orig = cv2.imread(path + im_id)

# Contrast enhancement
#img = np.zeros(img_orig.shape, img_orig.dtype)

alpha = 2
beta = 50
img = alpha*img_orig + beta
for y in range(img.shape[0]):
    for x in range(img.shape[1]):
        for c in range(img.shape[2]):
            img[y,x,c] = np.clip(img_orig[y,x,c]*alpha+beta,0,255) 
    print('row: ', y)
img_test = np.clip(img_orig*alpha+beta,0,255)
print('type: ', type(img_test[0,0,0]))
print('type img: ', type(img[0,0,0]))
print('equal: ', np.array_equal(img,img_test))
plt.imshow(img)
plt.show()
print(img.shape)
'''
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 
img_gray = cv2.GaussianBlur(img,(5,5),0)    # blurring
img_clahe = Clahe(img)    # Clahe

plt.figure(1)          
plt.imshow(img)
plt.title('contras enhancement')
plt.figure(2)
plt.imshow(img_gray)
plt.title('blurred')
plt.figure(3)
plt.imshow(img_clahe)
plt.title('Clahe')

edges_blurred = cv2.Canny(img,50,120)
edges_clahe = cv2.Canny(img_clahe,30,100)

plt.figure(4)
plt.subplot(131),plt.imshow(edges_blurred,cmap = 'gray')
plt.title('Original Image')
#plt.subplot(132),plt.imshow(edges2,cmap = 'gray')
#plt.title('New Image'), plt.xticks([]), plt.yticks([])
plt.subplot(133),plt.imshow(edges_clahe,cmap='gray')
plt.title('Clahe')


plt.show()
'''