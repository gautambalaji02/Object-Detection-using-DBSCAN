# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 15:29:58 2023

@author: Gautam Balaji
"""
#from sklearn.metrics._dist_metrics import ManhattanDistance 

# Directories
drawing_dir = 'Data/Input/Drawings/drawing_6.png'
main_template_dir = 'Data/Input/Templates/Template_Set_1/'
augmented_template_dir = 'Data/Input/Augmented/'
output_dir = 'Data/Output/'

# Template matching
method = "TM_CCORR_NORMED"
matched_thresh = 0.983
scale_range = [30, 150]
scale_interval = 1
rm_redundant = True
minmax = True
colors_list = ['red', 'blue', 'green', 'orange', 'brown', 'purple', 'yellow']

# DBSCAN
eps_main = 35
min_samples_main = 5

eps_cleanup = 20
min_samples_cleanup = 2
#metric = ManhattanDistance

# Random Crop
crop_scale = 0.55
num_crop_images  = 90
augment_split_morph = 0.67
augment_split_normal = 0.5

# Morphological Operations
threshold = 40
kernel_shape = 1
kernel_size = (3, 3)
iterations = 1
num_cyclic_morphing = 3

# Main
zscore_thres = 1.5
scale_thres = 8
mean_scale_thres = 4
std_dev_thres = 9
accuracy_thres = 0.01
