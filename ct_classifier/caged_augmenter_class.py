import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
import math
import random
from torchvision.transforms import functional as F
from torchvision.transforms import Pad

class CageAugmenter: 
    def __init__(self): # what stays fixed
        self.resolution = [1000, 1000]
        self.num_bars_x = 4
        self.num_bars_y = 6
        self.kernel = [30, 30]
        self.color_alpha = 0.3
    
    def __call__(self, img): # def rotatedRectWithMaxArea(self, w, h, angle):
        """
        Given a rectangle of size wxh that has been rotated by 'angle' (in
        radians), computes the width and height of the largest possible
        axis-aligned rectangle (maximal area) within the rotated rectangle.
        """
        if w <= 0 or h <= 0:
            return 0,0

        width_is_longer = w >= h
        side_long, side_short = (w,h) if width_is_longer else (h,w)

        # since the solutions for angle, -angle and 180-angle are all the same,
        # if suffices to look at the first quadrant and the absolute values of sin,cos:
        sin_a, cos_a = abs(math.sin(angle)), abs(math.cos(angle))
        if side_short <= 2.*sin_a*cos_a*side_long or abs(sin_a-cos_a) < 1e-10:
            # half constrained case: two crop corners touch the longer side,
            #   the other two corners are on the mid-line parallel to the longer line
            x = 0.5*side_short
            wr,hr = (x/sin_a,x/cos_a) if width_is_longer else (x/cos_a,x/sin_a)
        else:
            # fully constrained case: crop touches all 4 sides
            cos_2a = cos_a*cos_a - sin_a*sin_a
            wr,hr = (w*cos_a - h*sin_a)/cos_2a, (h*cos_a - w*sin_a)/cos_2a

        return int(wr),int(hr)
    

    def apply(self, im):
        self.im = im
        # Parameters that you randomize
        self.bar_radius_range = [10, 30]
        self.bar_radius = random.uniform(*self.bar_radius_range)

        self.random_color = [random.randint(0, 255) for _ in range(3)]
        self.mask_color = self.random_color

        self.rotate_range = [-20, 20]
        self.rotate = math.radians(random.uniform(*self.rotate_range))

            # How I want to call it
    rotatedRectWithMaxArea 
    augmenter = CageAugmenter(padding)
    img = augmenter (img)
    return img ?