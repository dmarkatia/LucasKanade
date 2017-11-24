"""
Danish Waheed
CAP5415 - Fall 2017

This Python program computes Lucas-Kanade optical flow without a Gaussian Pyramid
"""

import numpy as np
from matplotlib import pyplot as plt
import cv2
from LK import reduce, expand, compute_flow_map, lucas_kanade
np.seterr(divide='ignore', invalid='ignore')

# Setting the size of the window which will contain the flow map for both image sets
plt.figure(figsize=(15, 10))

# Reading in the first set of images
basketball1 = cv2.cvtColor(cv2.imread('basketball1.png'), cv2.COLOR_RGB2GRAY)
basketball2 = cv2.cvtColor(cv2.imread('basketball2.png'), cv2.COLOR_RGB2GRAY)

#Reading in the second set of images
grove1 = cv2.cvtColor(cv2.imread('grove1.png'), cv2.COLOR_RGB2GRAY)
grove2 = cv2.cvtColor(cv2.imread('grove2.png'), cv2.COLOR_RGB2GRAY)

u1, v1 = lucas_kanade(basketball1, basketball2)
u2, v2 = lucas_kanade(grove1, grove2)

flow_map1 = compute_flow_map(u1,v1,8)
flow_map2 = compute_flow_map(v1, v2, 8)

plt.imsave("basketball_flow_map.png", flow_map1, cmap='gray')
plt.imshow(flow_map1, cmap='gray')
plt.show()

plt.imsave("grove_flow_map.png", flow_map2, cmap='gray')
plt.imshow(flow_map2, cmap='gray')
plt.show()

# Basketball with the flow map overlay
image_mask1 = basketball1 + flow_map1
plt.imsave("basketball_with_mask.png", image_mask1, cmap='gray')
plt.imshow(image_mask1, cmap='gray')
plt.show()

# Grove image with the flow map overlay
image_mask2 = grove1 + flow_map2
plt.imsave("grove_with_mask.png", image_mask2, cmap='gray')
plt.imshow(image_mask2, cmap='gray')
plt.show()
