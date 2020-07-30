# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 13:58:30 2020

@author: Hemaxi
"""

'''Read a 3D image (saved in a .czi file), perform normalization (for each slice'''
'''along the z-direction), and save as a tif file to visualize on IMARIS'''


#import the necessary packages
from czifile import imread
import numpy as np
import os

# name of the images
imgs = ['Crop1', 'Crop2', 'Crop3', 'Crop4', 'Crop5_BC', 'Crop6_BC', 'Crop7_BC',
        'Crop8_BC', 'Low_P6_ICAM_A', 'P6_AVM_x40_CD31', 'P6_ICAM2', 'P6_ICAM2_SF',
        'P20_ICAM2_plexus']


#name of the image
index = 0   # <----- choose a number between 0 and 12
img_name = imgs[index]

## directory where the image is saved and directory to save the normalized image
img_dir = os.path.join(r'D:\iMM\data\Dataset\Images', img_name + '.czi')
save_dir = os.path.join(r'D:\iMM\data\Dataset\Normalized_Images', img_name + '.tif')


## read the image and pre-process
image = imread(img_dir)
image = image.squeeze()

image = image.transpose(0,2,3,1)  #C,X,Y,Z
uint16_max = 65535


## save the rgb order of the images
order = ['rgb', 'rgb', 'rgb', 'rgb', 'rgb', 'rgb', 'rgb',
        'rgb', 'grb', 'grb', 'gbr', 'gbr', 'grb']

rgb_order = order[index]  #depends on the image

image_r = ((np.expand_dims(image[rgb_order.index('r')],axis=0)/uint16_max)).astype(np.float32)
image_g = ((np.expand_dims(image[rgb_order.index('g')],axis=0)/uint16_max)).astype(np.float32)
image_b = np.zeros((1, image.shape[1], image.shape[2], image.shape[3])).astype(np.float32)
image = np.concatenate((image_r,image_g,image_b),axis=0)

image = image.transpose(1,2,3,0)   #X,Y,Z,C

# normalization for each 2D slice
def normalization(image):
  minn = np.min(image)
  maxx = np.max(image)
  ones = np.ones(np.shape(image))
  ones = ones * minn
  image_norm = (image-ones)/(maxx-minn)
  return image_norm

## perform the normalization for each 2D slice along the Z-direction
for k in range(0, np.shape(image)[2]):
    image[:,:,k,0] = normalization(image[:,:,k,0])
    image[:,:,k,1] = normalization(image[:,:,k,1])
    image[:,:,k,2] = normalization(image[:,:,k,2])
    

image = image*255.0
image = image.astype('uint8')

#write the image as .tif to visualize on IMARIS
from tifffile import imwrite
imwrite(save_dir, image, photometric='rgb')