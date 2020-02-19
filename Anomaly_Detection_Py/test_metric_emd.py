# test metric EMD 
import cv2
import numpy as np
import metric
import time

# path
path_normal = '/home/Meko/Repos/Master/Anomaly_Detection_Py/data/carpet/train/good'
path_anomal = '/home/Meko/Repos/Master/Anomaly_Detection_Py/data/carpet/test/hole'

img1 = cv2.imread(path_normal + '/000.png')
img2 = cv2.imread(path_normal + '/001.png')
img3 = cv2.imread(path_anomal+ '/000.png')
img4 = cv2.imread(path_anomal+ '/001.png')

img_size = img1.shape[:2]
IMG_SCALE = 0.25

img1 = cv2.resize(img1,tuple(int(IMG_SCALE * size) for size in img_size))
img2 = cv2.resize(img2,tuple(int(IMG_SCALE * size) for size in img_size))
img3 = cv2.resize(img3,tuple(int(IMG_SCALE * size) for size in img_size))
img4 = cv2.resize(img4,tuple(int(IMG_SCALE * size) for size in img_size))
img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
img3 = cv2.cvtColor(img3,cv2.COLOR_BGR2GRAY)
img4 = cv2.cvtColor(img4,cv2.COLOR_BGR2GRAY)


im_C = img1.shape[1]
im_R = img1.shape[0]

print('calculate cost mat')
cost_mat = metric.EMD_details.ground_distance_matrix_of_2dgrid(im_C,im_R)
print('Costmat calculated')
maxCost = metric.EMD_details.max_in_distance_matrix(cost_mat)
print('Construct Distance Object')
distance = metric.EMD(cost_mat, maxCost)
print('Distance Object calculated')

img1_reshaped = np.int_([])
img2_reshaped = np.int_([])
img3_reshaped = np.int_([])
img4_reshaped = np.int_([])
for i in range(0, im_R):
    for j in range(0, im_C):
        img1_reshaped = np.append(img1_reshaped, img1[i][j])
        img2_reshaped = np.append(img2_reshaped, img2[i][j])
        img3_reshaped = np.append(img3_reshaped, img3[i][j])
        img4_reshaped = np.append(img4_reshaped, img4[i][j])

print('shape {}, type {}, dtype {}'.format(img1_reshaped.shape,type(img1_reshaped),type(img1_reshaped[0])))
print('Calculate distances')
start_time = time.time()
dist12 = distance(img1_reshaped,img2_reshaped)
print('Distance 12 calculated')
dist13 = distance(img1_reshaped,img3_reshaped)
print('Distance 13 calculated')
dist34 = distance(img3_reshaped,img4_reshaped)
print('Distance 34 calculated')
end_time = time.time()        

print('dist12: {},\n dist13: {},\n dist34: {}'.format(dist12,dist13,dist34))
