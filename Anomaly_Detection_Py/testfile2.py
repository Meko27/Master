from glob import glob
import cv2
import numpy as np
from skimage.feature import hog
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from HOG_ground_dist import HOG_ground_dist
import time
from pdist_emd import pdist_split
import ray
import itertools

def chunked_iterable(iterable, size):
# Auxiliary function to iterate over chunks
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, size))
        if not chunk:
            break
        yield chunk



#path_normal = '/Users/meko/Documents/MATLAB/Anomaly_Detection/data/carpet/train/good' #MacOS
#path_normal = '/mnt/c/Users/welschm_2/Desktop/MK/Masterarbeit/data/carpet/train/good' #WSL
#path_normal = '/home/mk/Master/Anomaly_Detection_Py/data/carpet/train/good'
path_normal = '/home/Meko/Repos/Master/Anomaly_Detection_Py/data/carpet/train/good' #24core
n_images = 4

imgs = []
for i in range(n_images):
    if i<10:
        img_path = (path_normal + '/00{}.png'.format(i))
    elif i<100:
        img_path = (path_normal + '/0{}.png'.format(i))
    else:
        img_path = (path_normal + '/{}.png'.format(i))
    print('img_path: ', img_path )
    imgs.append(cv2.imread(img_path))

#img1 = cv2.imread(img2_path)
#img2 = cv2.imread(img1_path)
#imgs = [img1, img2]

img_size = imgs[0].shape[:2]

imgs_scaled = []

# scale images
img_scale = 0.3
for img in imgs:
    img_resized = cv2.resize(img,tuple(int(img_scale * size) for size in img_size))
    imgs_scaled.append(img_resized)
print('img resized to: ', imgs_scaled[0].shape)

# HOG parameters:
orientations = 9
pixels_per_cell = (64, 64)
cells_per_block = (4,4)
block_stride = tuple(int(cell_size -1) for cell_size in cells_per_block)
print('block_stride: ', block_stride)

hog_vect = []
for img_scaled in imgs_scaled:
    hog_feat = hog(img_scaled, orientations = orientations,
                pixels_per_cell = pixels_per_cell,
                cells_per_block = cells_per_block,multichannel=True)
    hog_vect.append(hog_feat)
hog_mat = np.array(hog_vect)

print('hog_size: ', hog_mat.shape)

# calculate EMD

ground_dist, d_hog = HOG_ground_dist(imgs_scaled[0],
                                    orientations=orientations,
                                    cell_size = pixels_per_cell,
                                    block_size = cells_per_block,
                                    block_stride = block_stride)
print('size grounddist: ', ground_dist.shape)


x1,x2 = pdist_split(hog_mat)
x1_t = np.float32(np.transpose(x1))
x2_t = np.float32(np.transpose(x2))
#emd_vect = np.empty(len(x1))
emd_vect = []

# parallel processing
ray.init(num_cpus=4)

@ray.remote
def calc_emd(x1,x2,grounddist):
    emd,_,_ = cv2.EMD(x1,x2,cv2.DIST_USER,cost=grounddist)
    return emd

emd = []
start = time.time()
'''
for i in range(len(x1)):
    #emd_vect[i],_,_ = cv2.EMD(x1_t[:,i],x2_t[:,i],cv2.DIST_USER, cost = ground_dist)
    emd.append(calc_emd.remote(x1_t[:,i],x2_t[:,i],ground_dist))

    print('calculate {} of {}'.format(i+1,len(x1)))
'''
end = time.time()
b = 0
start2 = time.time()
for batch in chunked_iterable(range(len(x1)), size=100):
    start = time.time()
    for i in batch:
        emd.append(calc_emd.remote(x1_t[:,i],x2_t[:,i],ground_dist))
    end = time.time()
    print('time elapsed {} batch {}: '.format(end-start,b+1))
    b += 1
    emd_vect = ray.get(emd)
end2 = time.time()

print('time elapsed ray.get', end2-start2)
print('emd: {}'.format(len(emd_vect)))
print('len(x1)' , len(x1))

print('----------------------------------------')
start3 = time.time()
emd_ref = calc_emd.remote(x1_t[:,1],x2_t[:,2],ground_dist)
emd_ref = ray.get(emd_ref)
end3 = time.time()
print('time: ',end3-start3)
print('hog:', emd_ref)
