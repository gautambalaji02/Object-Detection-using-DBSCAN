# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 16:21:30 2023

@author: GBALP935
"""

# Importing the libraries
import numpy as np
import cv2
from torchvision.transforms import RandomCrop
from PIL import Image
from Utils.Hyperparameters import *
from sklearn.model_selection import train_test_split

def RandomImageCrop(img, output_dir = augmented_template_dir):
   
    # Variables
    augmented_images = []
    
    # Converting the numpy array to a PIL image
    img_PIL = Image.fromarray(img)
    h, w = img_PIL.size 
    
    # Creating a randomcrop instance
    transform = RandomCrop(size=(int(h * crop_scale), int(w * crop_scale)))
                           
    # Generating 70 random cropped images

    for i in range(1,num_crop_images + 1):
        # Transformation
        output_PIL = transform(img_PIL)
        # PIL to numpy
        output = np.array(output_PIL)
        # Appending
        augmented_images.append(output)
        
        # Saving the augmented images
        file_dir = output_dir + 'augmented_sample_' + str(i) + '.png'
        cv2.imwrite(file_dir, output)
        
    # Performing a 'sharp-erode' split 
    augmented_images_set = train_test_split(augmented_images, train_size = augment_split_morph)
    augmented_images_sharp, augmented_images_erode = augmented_images_set[0], augmented_images_set[1] 
    
    # Performing the sharp normal split
    augmented_images_set = train_test_split(augmented_images_sharp, train_size = augment_split_normal)
    augmented_images_normal, augmented_images_sharp = augmented_images_set[0], augmented_images_set[1] 
    
    # Returning split images
    return augmented_images_sharp, augmented_images_erode, augmented_images_normal