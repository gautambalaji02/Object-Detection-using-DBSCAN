# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 13:57:08 2023

@author: Gautam Balaji
"""
# Libraries 
# Custom
from Utils.Random_Image_Crop import RandomImageCrop
from Utils.Template_Matching import invariantMatchTemplate
from Utils.Morphological_Operations import Morph_Operations
from Utils.Hyperparameters import *
# Inbuilt
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.stats import zscore
from tqdm import tqdm

def Processor(template_dir, drawing_dir = drawing_dir):
    # Algorithm
    
    # Necessary Local Variables
    point_info = []                      # To capture the red points and their info
    point_set = []                       # Capture red point coordinates
    scale_set = []                       # Scale of the symbol captured
    cluster_scale_set = []               # Capture the scales according to their cluster
    cluster_point_set = []               # Captures points according to their cluster
    cluster_probability_set = []         # Captures probability of every point in every cluster    
    num = 0                              # Alternating number and gen variable
    filtered_points = []                 # Points filtered/removed after refinement
    mean_points = []                     # Location of mean points of clusters
    mean_scale = []                      # mean scale of the centered points
    mean_sample_points = []              # How many sample points at the mean location            
    
    # Import the template and drawing images in grayscale
    template_img = cv2.imread(template_dir, 0)
    drawing_img = cv2.imread(drawing_dir, 0)
    
    '''
    The template image ideally should be taken from the drawing or search area for better results
    or otherwise use an image of appropriate size. Not too large or too small, and as close 
    as possible to the original size of the required fastener symbol
    '''
    # Perform Binarization and Morphological operations
    drawing_img_sharp, drawing_img_eroded, template_img_morphed = Morph_Operations(drawing_img, template_img)
    
    # Generate random crop images
    augmented_templates_sharp, augmented_templates_eroded, augmented_templates_normal = RandomImageCrop(template_img_morphed)
    
    # Obtain keypoints by template matching every cropped image
    print('\n---------Normal drawing template matching---------')
    for template in tqdm(augmented_templates_normal, colour = 'red'):
        # Removing unnecessary templates
        if template.all == 255:
            continue
        
        # Template matching
        points = invariantMatchTemplate(drawing_img, template, method, matched_thresh, scale_range, scale_interval, rm_redundant, minmax)
        for point in points:
            point_info.append(point)
            
    print('\n---------Sharp drawing template matching---------')
    for template in tqdm(augmented_templates_sharp, colour = 'green'):
        # Removing unnecessary templates
        if template.all == 255:
            continue
        
        # Template matching
        points = invariantMatchTemplate(drawing_img_sharp, template, method, matched_thresh, scale_range, scale_interval, rm_redundant, minmax)
        for point in points:
            point_info.append(point)
            
    print('\n---------Eroded drawing template matching---------')
    for template in tqdm(augmented_templates_eroded, colour = 'blue'):
        # Removing unnecessary templates
        if template.all == 255:
            continue
        
        # Template matching
        points = invariantMatchTemplate(drawing_img_eroded, template, method, matched_thresh, scale_range, scale_interval, rm_redundant, minmax)
        for point in points:
            point_info.append(point)
    
    # If any points are not detected   
    if len(point_info) == 0:
        print("There are no points, hence the fastener was not detected")   
        return None, drawing_img, template_img
        
    print('\n---------All the sample points detected---------')
    #Display(point_info, drawing_img, template_img)
    
    # Obtaining points from point info
    for info in point_info:
        # Extracting features
        point = list(info[0])
        scale = info[1]
        
        # Appending into separate sets
        scale_set.append(scale)
        point_set.append(point)
        
    # DBSCAN algorithm for fitting the points
    model = DBSCAN(eps = eps_main, min_samples = min_samples_main)
    model.fit(point_set) 
    print('\n---------Clusters Formed---------')
    
    # Grouping points into clusters
    for i in range(len(set(model.labels_))):
        # temp variables
        cluster_probability = []
        cluster_scale = []
        cluster_set = []
        
        for index in range(len(point_set)):
            if model.labels_[index] == i:
                # Extracting points of cluster
                cluster_probability.append(point_info[index][2])
                cluster_scale.append(scale_set[index])
                cluster_set.append(point_set[index]) 
                
        # Appending scale and points according to the cluster
        mean_sample_points.append(len(cluster_set))
        cluster_scale_set.append(cluster_scale)
        cluster_point_set.append(cluster_set)
        cluster_probability_set.append(cluster_probability)
        
    # Checking for empty sets
    for num_points in mean_sample_points:
        if num_points == 0:
            index = mean_sample_points.index(num_points)
            mean_sample_points.pop(index)
            cluster_point_set.pop(index)
            cluster_scale_set.pop(index)
            cluster_probability_set.pop(index)
            
    # Checking if any points exist
    if len(cluster_point_set) == 0:
        print("There are no points, hence the fastener was not detected")
        return None, drawing_img, template_img
    
    # Finding max probability from every cluster
    max_probability = [max(cluster) for cluster in cluster_probability_set]
    max_accuracy = max(max_probability)
     
    # Creating copies of the cluster scales, no. of samples and points before refinement
    cluster_point_set_copy = cluster_point_set.copy()
    cluster_scale_set_copy = cluster_scale_set.copy()
    mean_sample_points_copy = mean_sample_points.copy()   
    cluster_probability_set_copy = cluster_probability_set.copy()
    max_probability_copy = max_probability.copy()
    
    # Refinement of points
    print('\n---------Accuracy refinement of clusters---------')
    
    # Accuracy check
    for cluster in cluster_probability_set_copy:
        if max_accuracy - max(cluster) >= accuracy_thres:
            index = cluster_probability_set.index(cluster)
            filtered_points.append(index)
    
    # Sorting the indices in reverse order
    filtered_points.sort(reverse = True)
    
    # Removing unwanted clusters
    for i in filtered_points:
        cluster_point_set.pop(i)
        cluster_scale_set.pop(i)
        mean_sample_points.pop(i)    
        cluster_probability_set.pop(i)
        max_probability.pop(i)
        
    # Checking if any points exist
    if len(cluster_point_set) == 0:
        print("There are no points, hence the fastener was not detected")
        return None, drawing_img, template_img
    else:
        # no. of clusters removed
        num = len(cluster_probability_set_copy) - len(cluster_probability_set)
        print(f'Number of Clusters removed after accuracy check (accuracy threshold: {accuracy_thres}): {num}')
    
    # Standard deviation check
    print('\n---------Scale Variance refinement of clusters---------')
    
    # Calculating z-score to remove outliers
    for i in range(len(cluster_scale_set)):
        #print(i)
        Z_Score = zscore(cluster_scale_set[i])
        filtered_points = [j for j in range(len(Z_Score)) if abs(Z_Score[j]) >= zscore_thres]
        
        # Sorting the indices in reverse order
        filtered_points.sort(reverse = True)
        
        for j in filtered_points:
            #print(filtered_points)
            cluster_scale_set[i].pop(j)
            cluster_point_set[i].pop(j)
            cluster_probability_set[i].pop(j)
            mean_sample_points[i] -= 1
    
    # Calculating standard deviation
    filtered_points = []
    for cluster in cluster_scale_set:
        # Working with a copy
        clus = cluster.copy()
        clus = np.array(clus)
        
        # Finding the standard deviation of the cluster
        std = np.std(clus)
        
        # Condition
        if std >= std_dev_thres:
            #Position of that undesirable cluster
            index = cluster_scale_set.index(cluster)
            filtered_points.append(index)     
            
    # Sorting the indices in reverse order
    filtered_points.sort(reverse = True)
    
    # Removing unwanted clusters
    for i in filtered_points:
        cluster_point_set.pop(i)
        cluster_scale_set.pop(i)
        mean_sample_points.pop(i)    
        cluster_probability_set.pop(i)    
        max_probability.pop(i)
        
     # Checking if any points exist
    if len(cluster_point_set) == 0:
        print("There are no points, hence the fastener was not detected")
    else:
        # no. of clusters removed
        num = len(cluster_probability_set_copy) - len(cluster_probability_set)
        print(f'Number of Clusters removed after scale variance check (scale threshold: {scale_thres}): {num}')
              
    # Finding the mean point, scale, and probability of the clusters
    # Finding the mean points
    for cluster in cluster_point_set:
        Sum = np.array([0, 0])
        for point in cluster:
            Sum += np.array(point)
        #print(Sum)
        num = len(cluster) 
        if not num == 0:
            array = np.array([int(Sum[0] / num), int(Sum[1] / num)])
            mean_points.append(array)
    
    # Finding the mean scale of every cluster
    for cluster in cluster_scale_set:
        Sum = 0
        for scale in cluster:
            Sum += scale
        #print(Sum)
        num = len(cluster) 
        if not num == 0:
            scale = int(Sum / num)
            mean_scale.append(scale)
        
    # Copy of the 'mean' variables
    mean_points_copy = mean_points.copy()
    mean_scale_copy = mean_scale.copy()
    
    print('\n---------Mean scale refinement of clusters---------')
    # Further refinement of points
    filtered_points = []
    
    # Find the scale of point with the highest accuracy
    true_scale = mean_scale[max_probability.index(max_accuracy)]
    
    # Conditions
    for i in range(len(mean_scale)):
        if max_probability[i] > 0.99:
            continue
        elif abs(mean_scale[i] - true_scale) >= mean_scale_thres:
            filtered_points.append(i)
                          
    # Sorting the indices in reverse order
    filtered_points.sort(reverse = True)
    
    # Removing unwanted mean points
    for i in filtered_points:
        mean_points.pop(i)
        mean_scale.pop(i)
        mean_sample_points.pop(i)   
        max_probability.pop(i)    
        
    # Number of points removed
    num = len(mean_points_copy) - len(mean_points)
    print(f'Number of mean points removed after mean scale check (mean scale threshold: {mean_scale_thres}): {num}')
    
    # Conversion to numpy arrays
    mean_points = np.array(mean_points) 
    mean_scale = np.array(mean_scale).reshape((len(mean_scale),1))
    max_probability = np.array(max_probability).reshape((len(max_probability),1))
    mean_sample_points = np.array(mean_sample_points).reshape((len(mean_sample_points),1))
    
    # Mean point info
    '''
    Format of input: (coordinates, sample_points, scale, cross_corr_coeff)
    '''
    mean_points_info = np.concatenate((mean_points, mean_sample_points, mean_scale, max_probability), axis = 1)
    
    print('\n---------Returning the refined points---------')
    # Displaying the points after refinement
    #Display(mean_points_info, drawing_img, template_img)
    return mean_points_info, drawing_img, template_img