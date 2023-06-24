# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 12:39:20 2023

@author: Gautam Balaji
"""

# Libraries
# Custom
from Utils.Hyperparameters import main_template_dir, drawing_dir
from Processor import Processor
from Cleanup import Cleanup
from Utils.Template_Matcher import Display
# Inbuilt
import os

# Variables
Points = {}             # Dictionary to store all points information
Template_img_list = []  # List of template images

# Algorithm
# Obtaining the list of templates to work with
template_list = os.listdir(main_template_dir)

# Getting the template IDs
Template_IDs = [name.split('.')[0] for name in template_list]

# Passing the templates through the processor
for i in range(len(template_list)):
    # Processing
    template_dir = main_template_dir + template_list[i]
    Points[Template_IDs[i]], drawing, template = Processor(template_dir, drawing_dir)
    Template_img_list.append(template)
 
# Cleanup operation
Points = Cleanup(Points)

# Displaying the Points
Display(Points, drawing, Template_img_list)
