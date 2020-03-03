# Image Preprocessing

import cv2 
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
from skimage.filters import threshold_otsu
from scipy import ndimage
from PIL import Image
import time
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
img = cv2.imread(path + im_id)

img_g = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) # to gray
print('shape: ', img_g.shape)

####### contrast scretching######
# Create zeros array to store the stretched image
img_cont1 = np.zeros((img.shape[0],img.shape[1]),dtype = 'uint8')
img_cont2 = img_cont1

# Loop over the image and apply Min-Max formulae
#min_i = np.min(img_g)
#max_i = np.max(img_g)
#max_min = max_i-min_i
#start1 = time.time()
#for i in range(img_g.shape[0]):
#    for j in range(img_g.shape[1]):
#        img_cont1[i,j] = 255*(img_g[i,j]-min_i)/max_min
#print('loop2')
#end1 = time.time()
#print('loop1: ', end1-start1)
# Loop over the image and apply Min-Max formulae
alpha = 2
beta = 20
start2 = time.time()
for i in range(img_g.shape[0]):
    for j in range(img_g.shape[1]):
       img_cont2[i,j] = np.clip(alpha*img_g[i,j]+beta,0,255)

end2 = time.time()
print('loop2: ', end2-start2)
    
# display input and both output images
plt.figure(1)
plt.imshow(img_g,cmap='gray')
plt.title('orgimg')
#plt.figure(2)
#plt.imshow(img_cont1,cmap='gray')
#plt.title('norm1')
#plt.figure(3)
#plt.imshow(img_cont2,cmap='gray')
#plt.title('norm2')
#plt.show()

######EDGE DETECTION######

edges = cv2.Canny(img_cont2,70,100,apertureSize = 3)
kernel = np.array([[0,0,0,0,0,0,1,2,1,0,0,0,0,0,0], 
                    [0,0,0,0,0,0,1,2,1,0,0,0,0,0,0], 
                    [0,0,0,0,0,0,1,2,1,0,0,0,0,0,0], 
                    [0,0,0,0,0,0,1,2,1,0,0,0,0,0,0], 
                    [0,0,0,0,0,0,1,2,1,0,0,0,0,0,0],
                    [0,0,0,0,0,0,1,2,1,0,0,0,0,0,0],
                    [0,0,0,0,0,0,1,0,1,0,0,0,0,0,0],
                    [0,0,0,0,0,0,1,2,1,0,0,0,0,0,0],
                    [0,0,0,0,0,0,1,2,1,0,0,0,0,0,0],
                    [0,0,0,0,0,0,1,2,1,0,0,0,0,0,0],
                    [0,0,0,0,0,0,1,2,1,0,0,0,0,0,0],
                    [0,0,0,0,0,0,1,2,1,0,0,0,0,0,0],
                    [0,0,0,0,0,0,1,2,1,0,0,0,0,0,0]])

kernel = kernel[2:-2,5:10]
kernel_r = kernel.shape[0]
kernel_c = kernel.shape[1]
edges2 = np.zeros(img_g.shape)
for i in range((kernel_r-1)/2,img_g.shape[0]-(kernel_r-1)/2 -1):
    for j in range((kernel_c-1)/2,img_g.shape[1]-(kernel_c-1)/2 -1):
        end_j=j + ((kernel_c-1)/2 +1)
        start_j = j-(kernel_c-1)/2
        local_pixels = edges[i-(kernel_r-1)/2:i+(kernel_r-1)/2 + 1 , start_j : end_j]
        result = local_pixels * kernel
        a=np.sum(result)
        edges2[i,j] = np.sum(result)
        if edges2[i,j] < 1500:
            edges2[i,j] = 0


#####Detect Object####

# Hough transform
rho = 2
theta = np.pi/360   
threshold = 1000
lines = cv2.HoughLines(edges2,rho,theta,threshold)

for rho,theta in lines[0]:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))

    cv2.line(img_g,(x1,y1),(x2,y2),(0,0,255),2)

cv2.imwrite('lines.jpg',img_g)
figure(2)
plt.imshow(img_g)


plt.figure(3)
plt.imshow(edges2,cmap='gray')
plt.title('edges2')

plt.figure(4)
plt.imshow(edges,cmap='gray')
#plt.figure(3)
#plt.imshow(lines)
plt.show()