'''
Function to calculate the ground distance function of a HOG feature descriptor
'''
import numpy as np
import cv2
from scipy.spatial.distance import pdist, squareform

def HOG_ground_dist(img,orientations=9,
                    cell_size = (8,8),
                    block_size = (2,2),
                    block_stride = (1,1),
                    signed = 0,
                    rotation_cost = 1,
                    move_cost = 1,
                    threshold = 0):
    img_size = img.shape[:2]
    print('img_size: ', img_size)
    blocks_per_img = np.floor((img_size/np.array(cell_size)-np.array(block_size))/(np.array(block_size)-np.array(block_stride))+1)    
    def multiply(array):
        result = 1
        for i in array:
            result = result * i
        return result
    n_hog_bins = multiply(blocks_per_img) * multiply(block_size) * orientations
    # spatial distance
    cell_i_vect = np.zeros((int(n_hog_bins/orientations),1))
    cell_j_vect = cell_i_vect
    idx = 0
    for b_i in range(int(blocks_per_img[0])):
        for b_j in range(int(blocks_per_img[1])):
            for cb_j in range(int(block_size[1])):
                for cb_i in range(int(block_size[0])):
                    if b_j==1:
                        cell_j = cb_j
                    else:
                        cell_j = cb_j + (b_j - 1)
                    if b_i == 1:
                        cell_i = cb_i
                    else:
                        cell_i = cb_i + (b_i - 1)
                    cell_i_vect[idx] = cell_i
                    cell_j_vect[idx] = cell_j
                    idx += 1
    
    cell_idx_pair_vect = np.concatenate((cell_i_vect,cell_j_vect),axis=1)
    spatial_dist_mat = squareform(pdist(cell_idx_pair_vect,'cityblock'))
    
    if threshold != 0:
        spatial_dist_mat[spatial_dist_mat>0] = threshold
    spatial_dist_mat = np.repeat(spatial_dist_mat,orientations,axis=0)
    spatial_dist_mat = np.repeat(spatial_dist_mat,orientations,axis=1)

    # rotation distance
    def angle_diff(a,b,max_angle):
        norm_deg = np.mod(a-b,max_angle)
        abs_diff_deg = np.minimum(max_angle-norm_deg,norm_deg)
        return abs_diff_deg

    if signed == 1:
        max_angle = 360
    else:
        max_angle = 180
    orients_vect = np.linspace(0,max_angle*(1-1/orientations),orientations)
    orients_dist_vect = []
    idx = 0
    for i in range(orientations-1):
        start = i+1
        for j in range(start,orientations):
            orients_dist_vect.append(angle_diff(orients_vect[i],orients_vect[j],max_angle))
            idx +=1
    angle_unit_cost = 20
    diff_mat = squareform(np.array(orients_dist_vect))
    orient_dist_cell = diff_mat/angle_unit_cost
    orient_dist_mat = np.tile(orient_dist_cell,(int(n_hog_bins/orientations),int(n_hog_bins/orientations)))

    # total ground distance matrix
    ground_dist = np.float32(rotation_cost * orient_dist_mat + move_cost * spatial_dist_mat)

    return ground_dist,n_hog_bins