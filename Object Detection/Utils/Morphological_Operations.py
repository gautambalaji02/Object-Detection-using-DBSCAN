# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 12:00:37 2023

@author: Gautam Balaji
url: https://docs.opencv.org/3.4/d4/d86/group__imgproc__filter.html#ga7be549266bad7b2e6a04db49827f9f32
"""
# Import libraries
import cv2
import numpy as np
from Utils.Hyperparameters import *

# Function
def Morph_Operations(drawing_copy, template_copy, threshold = threshold, kernel_shape = kernel_shape, kernel_size = kernel_size, iterations = iterations):
    # Copies of the template and drawing
    template = np.copy(template_copy)
    drawing = np.copy(drawing_copy)
    
    # Perform a thresholding operation on drawing and template - otsu thresholding
    _, drawing_bin = cv2.threshold(drawing, threshold, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, template_bin = cv2.threshold(template, threshold, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Displaying the template and the drawing
    cv2.imshow('otsu drawing', drawing_bin)
    cv2.imshow('otsu template', template_bin)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Morphological operations
    element = cv2.getStructuringElement(kernel_shape, kernel_size)
    
    # operation on template - closing
    template_morphed = cv2.morphologyEx(template_bin, cv2.MORPH_OPEN, element, iterations = iterations)
    
    # operation on drawing - several opening-closing
    drawing_morphed = cv2.morphologyEx(drawing_bin, cv2.MORPH_ERODE, element, iterations = iterations)    
    
    for i in range(num_cyclic_morphing):
        drawing_morphed = cv2.morphologyEx(drawing_morphed, cv2.MORPH_OPEN, element, iterations = iterations)
        drawing_morphed = cv2.morphologyEx(drawing_morphed, cv2.MORPH_CLOSE, element, iterations = iterations)
    
    drawing_morphed_sharp = cv2.morphologyEx(drawing_morphed, cv2.MORPH_DILATE, element, iterations = iterations)
    drawing_morphed_eroded = cv2.morphologyEx(drawing_morphed, cv2.MORPH_DILATE, element, iterations = iterations * 2)
    
    # Displaying the template and the drawing
    cv2.imshow('morphed drawing sharp', drawing_morphed_sharp)
    cv2.imshow('morphed drawing eroded', drawing_morphed_eroded)
    cv2.imshow('normal drawing', drawing_copy)
    cv2.imshow('morphed template', template_morphed)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Returning
    return drawing_morphed_sharp, drawing_morphed_eroded, template_morphed
