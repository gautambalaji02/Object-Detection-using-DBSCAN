# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 10:12:32 2023

@author: GBALP935
"""
# Libraries
# Custom
from Utils.Hyperparameters import eps_cleanup, min_samples_cleanup

# Inbuilt
from sklearn.cluster import DBSCAN
import numpy as np

'''
Format of input: Symbol ID, (coordinates, sample_points, scale, cross_corr_coeff)
'''
# Determines the most probable fastener among similar fastener symbols
def Cleanup(Points):
    print('\n---------Cleanup started---------')
    
    # Variables
    point_set = []                          # Coordinates of all the points detected
    cluster_coordinates_set = []            # Coordinates of points in clusters
    cluster_samples_set = []                # No. of samples for every point
    cluster_index_set = []                  # To store the indices of points in clusters
    Cleaned_Points = {}                     # Dictionary to store the new point set
    unwanted_points_index = []              # List to store the indices of the points to be removed
    
    #IDs of the fasteners
    Keys = list(Points.keys())
    
    # Creating the point coordinates set
    for i in range(len(Points)):
        key = Keys[i]
        empty = list(Points.values())[i]
        if type(empty) != np.ndarray:
            continue
        for point_info in list(Points.values())[i]:
            point = []
            point.append(key)
            point.extend(point_info)
            point_set.append(point)
    
    # Clustering with DBSCANg
    model = DBSCAN(eps = eps_cleanup, min_samples = min_samples_cleanup)
    model.fit([point[1:3] for point in point_set])
    #print(model.labels_)
    
    # If there are no clusters formed
    if (len(set(model.labels_)) == 1) and (model.labels_[0] == -1):
        print('There are no overlapping points detected')
        return Points
    
    # Finding the clusters
    for i in range(len(set(model.labels_))):
        cluster_sample = []
        cluster_point = []
        cluster_index = []
        
        for index in range(len(point_set) - 1):
            # Condition
            if model.labels_[index] == i:
                # Extracting points info of clusters
                cluster_index.append(index)
                cluster_point.append(point_set[index][1:3]) 
                cluster_sample.append(point_set[index][3]) 
                
        # Appending the clusters and its info to the sets
        cluster_samples_set.append(cluster_sample)
        cluster_coordinates_set.append(cluster_point)
        cluster_index_set.append(cluster_index)
        
    # Checking for empty sets
    for cluster in cluster_index_set:
        if len(cluster) == 0:
            index = cluster_index_set.index(cluster)
            cluster_coordinates_set.pop(index)
            cluster_index_set.pop(index)
            cluster_samples_set.pop(index)
            
    # Comparing the no. of sample points in every cluster
    for i in range(len(cluster_samples_set)):
        max_samples_index = cluster_samples_set[i].index(max(cluster_samples_set[i]))
        print('point to be retained: ', cluster_coordinates_set[i][max_samples_index])
        
        cluster_index_set[i].pop(max_samples_index)
        cluster_index_set[i].reverse()
        
        # Collecting the unwanted points from the Points list
        for index in cluster_index_set[i]:
            unwanted_points_index.append(index)
    
    # Removing the unwanted points
    unwanted_points_index.sort(reverse = True)
    for index in unwanted_points_index:
        point_set.pop(index)
            
    # Appending the new set
    for key in Keys:
        cluster_points = []
        for i in range(len(point_set)):
            if key == point_set[i][0]:
                cluster_points.append(point_set[i][1:])
        Cleaned_Points[key] = np.array(cluster_points)
    
    # Returning the Cleaned set
    return Cleaned_Points                             