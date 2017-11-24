"""
Danish Waheed
CAP5415 - Fall 2017

This Python program computes Lucas-Kanade optical flow with a Gaussian Pyramid
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt
from LK import reduce, expand, lucas_kanade, compute_flow_map


# Setting the size of the window which will contain the before and after
# images of both image sets, along with their flow maps
plt.figure(figsize=(15, 10))

# Reading in the first set of images
basketball1 = cv2.cvtColor(cv2.imread('basketball1.png'), cv2.COLOR_RGB2GRAY)
basketball2 = cv2.cvtColor(cv2.imread('basketball2.png'), cv2.COLOR_RGB2GRAY)

# Reading in the second set of images
grove1 = cv2.cvtColor(cv2.imread('grove1.png'), cv2.COLOR_RGB2GRAY)
grove2 = cv2.cvtColor(cv2.imread('grove2.png'), cv2.COLOR_RGB2GRAY)


# We set num_levels to the number of levels which we want for our Gaussian pyramid
num_levels = 4

# Declaring a current_levels variable to keep track of the current level as we iterate through the pyramid
current_level = num_levels

while current_level > 0:
    b1 = reduce(basketball1, current_level)
    b2 = reduce(basketball2, current_level)

    if current_level == num_levels:
        u = np.zeros(b1.shape)
        v = np.zeros(b1.shape)
    else:
        u = 2 * expand(u)
        v = 2 * expand(v)

    dx, dy = lucas_kanade(b1, b2)

    u = u + np.uint8(dx)
    v = v + np.uint8(dy)

    flow_map = compute_flow_map(u, v, 8)
    plt.subplot(num_levels + 1, 3, 3 * (num_levels - current_level) + 1), plt.imshow(b1, cmap='gray')
    plt.subplot(num_levels + 1, 3, 3 * (num_levels - current_level) + 2), plt.imshow(flow_map, cmap='gray')
    plt.subplot(num_levels + 1, 3, 3 * (num_levels - current_level) + 3), plt.imshow(b2, cmap='gray')

    current_level -= 1

plt.savefig("basketball_pyramid.png")
plt.show()


# Re-declaring the current_level variable as it was decremented to zero for the first image
current_level = num_levels

while current_level > 0:
    g1 = reduce(grove1, current_level)
    g2 = reduce(grove2, current_level)

    if current_level == num_levels:
        u = np.zeros(g1.shape)
        v = np.zeros(g2.shape)
    else:
        u = 2 * expand(u)
        v = 2 * expand(v)

    dx, dy = lucas_kanade(g1, g2)

    u = u + np.uint8(dx)
    v = v + np.uint8(dy)

    flow_map = compute_flow_map(u, v, 8)
    plt.subplot(num_levels + 1, 3, 3 * (num_levels - current_level) + 1), plt.imshow(g1, cmap='gray')
    plt.subplot(num_levels + 1, 3, 3 * (num_levels - current_level) + 2), plt.imshow(flow_map, cmap='gray')
    plt.subplot(num_levels + 1, 3, 3 * (num_levels - current_level) + 3), plt.imshow(g2, cmap='gray')

    current_level -= 1

plt.savefig("grove_pyramid.png")
plt.title("Grove Pyramid")
plt.show()
