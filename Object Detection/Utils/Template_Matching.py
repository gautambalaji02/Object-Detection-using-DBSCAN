# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 12:22:55 2023

@author: Gautam Balaji
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from Utils.Hyperparameters import crop_scale, output_dir, colors_list


def scale_image(image, percent, maxwh):
    max_width = maxwh[1]
    max_height = maxwh[0]
    max_percent_width = max_width / image.shape[1] * 100
    max_percent_height = max_height / image.shape[0] * 100
    max_percent = 0
    if max_percent_width < max_percent_height:
        max_percent = max_percent_width
    else:
        max_percent = max_percent_height
    if percent > max_percent:
        percent = max_percent
    width = int(image.shape[1] * percent / 100)
    height = int(image.shape[0] * percent / 100)
    result = cv2.resize(image, (width, height), interpolation = cv2.INTER_AREA)
    return result, percent


def invariantMatchTemplate(grayimage, graytemplate, method, matched_thresh, scale_range, scale_interval, rm_redundant, minmax):
    """
    grayimage: gray image where the search is running.
    graytemplate: gray searched template. It must be not greater than the source image and have the same data type.
    method: [String] Parameter specifying the comparison method
    matched_thresh: [Float] Setting threshold of matched results(0~1).
    scale_range: [Integer] Array of range of scaling in percentage. Example: [50,200]
    scale_interval: [Integer] Interval of traversing the range of scaling in percentage.
    rm_redundant: [Boolean] Option for removing redundant matched results based on the width and height of the template.
    minmax:[Boolean] Option for finding points with minimum/maximum value.

    Returns: List of satisfied matched points in format [[point.x, point.y], scale].
    """
    grayimage = grayimage.astype(graytemplate.dtype)
    image_maxwh = grayimage.shape
    height, width = graytemplate.shape
    all_points = []
    if minmax == False:
        for next_scale in range(scale_range[0], scale_range[1], scale_interval):
            scaled_template, actual_scale = scale_image(graytemplate, next_scale, image_maxwh)
            if method == "TM_CCOEFF":
                matched_points = cv2.matchTemplate(grayimage,scaled_template,cv2.TM_CCOEFF)
                satisfied_points = np.where(matched_points >= matched_thresh)
            elif method == "TM_CCOEFF_NORMED":
                matched_points = cv2.matchTemplate(grayimage,scaled_template,cv2.TM_CCOEFF_NORMED)
                satisfied_points = np.where(matched_points >= matched_thresh)
            elif method == "TM_CCORR":
                matched_points = cv2.matchTemplate(grayimage,scaled_template,cv2.TM_CCORR)
                satisfied_points = np.where(matched_points >= matched_thresh)
            elif method == "TM_CCORR_NORMED":
                matched_points = cv2.matchTemplate(grayimage,scaled_template,cv2.TM_CCORR_NORMED)
                satisfied_points = np.where(matched_points >= matched_thresh)
            elif method == "TM_SQDIFF":
                matched_points = cv2.matchTemplate(grayimage,scaled_template,cv2.TM_SQDIFF)
                satisfied_points = np.where(matched_points <= matched_thresh)
            elif method == "TM_SQDIFF_NORMED":
                matched_points = cv2.matchTemplate(grayimage,scaled_template,cv2.TM_SQDIFF_NORMED)
                satisfied_points = np.where(matched_points <= matched_thresh)
                
            else:
                raise NotImplementedError("There's no such comparison method for template matching.")
            
            for pt in zip(*satisfied_points[::1]):
                all_points.append([pt, actual_scale])
    else:
        for next_scale in range(scale_range[0], scale_range[1], scale_interval):
            scaled_template, actual_scale = scale_image(graytemplate, next_scale, image_maxwh)
            if method == "TM_CCOEFF":
                matched_points = cv2.matchTemplate(grayimage,scaled_template,cv2.TM_CCOEFF)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(matched_points)
                if max_val >= matched_thresh:
                    all_points.append([max_loc, actual_scale, max_val])
            elif method == "TM_CCOEFF_NORMED":
                matched_points = cv2.matchTemplate(grayimage, scaled_template,cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(matched_points)
                if max_val >= matched_thresh:
                    all_points.append([max_loc, actual_scale, max_val])
            elif method == "TM_CCORR":
                matched_points = cv2.matchTemplate(grayimage,scaled_template,cv2.TM_CCORR)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(matched_points)
                if max_val >= matched_thresh:
                    all_points.append([max_loc, actual_scale, max_val])
            elif method == "TM_CCORR_NORMED":
                matched_points = cv2.matchTemplate(grayimage,scaled_template,cv2.TM_CCORR_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(matched_points)
                if max_val >= matched_thresh:
                    all_points.append([max_loc, actual_scale, max_val])
            elif method == "TM_SQDIFF":
                matched_points = cv2.matchTemplate(grayimage,scaled_template,cv2.TM_SQDIFF)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(matched_points)
                if min_val <= matched_thresh:
                    all_points.append([min_loc, actual_scale, min_val])
            elif method == "TM_SQDIFF_NORMED":
                matched_points = cv2.matchTemplate(grayimage,scaled_template,cv2.TM_SQDIFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(matched_points)
                if min_val <= matched_thresh:
                    all_points.append([min_loc, actual_scale, min_val])
            else:
                raise  NotImplementedError("There's no such comparison method for template matching.")
        if method == "TM_CCOEFF":
            all_points = sorted(all_points, key=lambda x: -x[2])
        elif method == "TM_CCOEFF_NORMED":
            all_points = sorted(all_points, key=lambda x: -x[2])
        elif method == "TM_CCORR":
            all_points = sorted(all_points, key=lambda x: -x[2])
        elif method == "TM_CCORR_NORMED":
            all_points = sorted(all_points, key=lambda x: -x[2])
        elif method == "TM_SQDIFF":
            all_points = sorted(all_points, key=lambda x: x[2])
        elif method == "TM_SQDIFF_NORMED":
            all_points = sorted(all_points, key=lambda x: x[2])
    if rm_redundant == True:
        lone_points_list = []
        visited_points_list = []
        for point_info in all_points:
            point = point_info[0]
            scale = point_info[1]
            all_visited_points_not_close = True
            if len(visited_points_list) != 0:
                for visited_point in visited_points_list:
                    if ((abs(visited_point[0] - point[0]) < (width * scale / 100)) and (abs(visited_point[1] - point[1]) < (height * scale / 100))):
                        all_visited_points_not_close = False
                if all_visited_points_not_close == True:
                    lone_points_list.append(point_info)
                    visited_points_list.append(point)
            else:
                lone_points_list.append(point_info)
                visited_points_list.append(point)
        points_list = lone_points_list
    else:
        points_list = all_points
    return points_list

    
def Display(Points, drawing_img, template_img_list):
    
    # Necessary Variables
    height, width = drawing_img.shape
    centers_list = []
    Keys = list(Points.keys())
    template_size = [template.shape for template in template_img_list]
    name = "final_"
    Color_Keys = {}
    colors_iter = iter(colors_list)
    size_iter = iter(template_size)
    
    for key in Keys:
        Color_Keys[key] = next(colors_iter)
    
    # For the bounding box, top left point and center point
    name2 = "boxes.png"
    file_name = output_dir + name + name2
    fig, ax = plt.subplots(1)
    plt.gcf()
    ax.imshow(drawing_img, cmap = 'binary_r')
    
    # Iterating through every fastener
    for key in Keys:
        # Another iterator
        size = next(iter(size_iter))
        fastener_point_info = Points.get(key)
        
        # Failsafe if there are no points
        if type(fastener_point_info) != np.ndarray:
            continue
        
        # For all the points in the list
        for point_info in fastener_point_info:
            # Fastener type
            print('\nFastener type: ', key)
            #coordinates
            point = [point_info[0], point_info[1]]
            print("Point:", point)
            # No. of samples
            sample_points = point_info[2]
            print("No. of sample points:", sample_points)
            # Scale
            scale = point_info[3]
            print("Corresponding scale:", scale)
            # Cross corr coeff
            probab = point_info[4]
            print("Probability of match:", probab)
            # Centers list
            centers_list.append([key, point, scale])
            
            plt.scatter(point[0], point[1], s=10, color=Color_Keys[key], label = key)
            rectangle = patches.Rectangle((point[0], point[1]), size[0]*crop_scale*scale/100, size[1]*crop_scale*scale/100, alpha = 0.50)
            ax.add_patch(rectangle)
    
    plt.legend(labels = Keys, bbox_to_anchor = (0.25, 1.15))   
    plt.show()
    fig.savefig(file_name)
    
    # Just the red points on the image
    name2 = "points.png"
    file_name = output_dir + name + name2
    fig2, ax2 = plt.subplots(1)
    plt.gcf()
    ax2.imshow(drawing_img, cmap = 'binary_r')
    for point_info in centers_list:
        key = point_info[0]
        point = point_info[1]
        scale = point_info[2]
        plt.scatter(point[0]+size[0]/2*crop_scale*scale/100, point[1]+size[1]/2*crop_scale*scale/100, c = Color_Keys[key], s=20, label=key)
    plt.legend(labels = Keys, bbox_to_anchor = (0.75, 1.15))
    plt.show()
    fig2.savefig(file_name)


'''
note: since the prediction is nto great with just one image of the symbol,
we shall try to perform a smart random crop to take out the features from the symbol
and perform the detection. Find the red points: perform a clustering algorithm-DBSCAN
and find the avg-mean position of the cluster. This should give all the points with symbol
'''