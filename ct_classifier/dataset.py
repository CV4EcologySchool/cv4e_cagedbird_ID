'''
    PyTorch dataset class for COCO-CT-formatted datasets. Note that you could
    use the official PyTorch MS-COCO wrappers:
    https://pytorch.org/vision/master/generated/torchvision.datasets.CocoDetection.html

    We just hack our way through the COCO JSON files here for demonstration
    purposes.

    See also the MS-COCO format on the official Web page:
    https://cocodataset.org/#format-data

    2022 Benjamin Kellenberger
'''

import os
import json
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, ToTensor, Lambda, Pad, functional, RandomHorizontalFlip,RandomAdjustSharpness, GaussianBlur, RandomVerticalFlip
from PIL import Image
import math
import matplotlib.pyplot as plt
import numpy as np
import cv2
import math
import random

def rotatedRectWithMaxArea(w, h, angle):
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

class CTDataset(Dataset):

    def __init__(self, cfg, split='train2'): # Not sure why it says train2
        '''
            Constructor. Here, we collect and index the dataset inputs and
            labels.
        '''
        self.data_root = cfg['data_root']

        # Add a test_root attribute to the CTDataset class, which points to the test set directory
        self.test_root = cfg.get('test_root', None)  # Add support for test_root

        self.split = split

        class FixedHeightResize:
            def __init__(self, size):
                self.size = size
                
            def __call__(self, img):
                w,h = img.size
                aspect_ratio = float(h) / float(w)
                if h>w:
                    new_w = math.ceil(self.size / aspect_ratio)
                    img = functional.resize(img, (self.size, new_w))
                else:
                    new_h = math.ceil( aspect_ratio * self.size) # it eas / before diving
                    img = functional.resize(img, (new_h,self.size))

                #c, h, w = img.shape
                w,h = img.size # PIL image formats are in w and h, transformed to rgb, h, w later, and needs to see size, Tensors # are seen in shape
                # a list can also be referred to as a sequence (same for a tuple)
                pad_diff_h = self.size - h 
                # print ("Print the pad_d_h")
                # print(pad_diff_h)

                pad_diff_w =self.size - w

                # print ("Print the pad_d_w")
                # print(pad_diff_w)
                
                padding = [0, pad_diff_h, pad_diff_w, 0]
                padder = Pad(padding)
                img = padder(img)

                return img

        # class CageAugmenter: 
        #     def __init__(self): # what stays fixed - but we don't add size? since we will copy paste images, do I nee
        #         # to list the self.resolution
        #         self.resolution = [1000, 1000]
        #         self.num_bars_x = 4
        #         self.num_bars_y = 6
        #         self.kernel = [30, 30]
        #         self.color_alpha = 0.3

        #     def __call__(self, img):
        #         # #set parameters
        #         # resolution = [1000, 1000]
        #         # num_bars_x = 4
        #         # num_bars_y = 6
        #         # kernel = [30, 30]
        #         # mask_color = [181, 148, 16]
        #         # color_alpha = .3

        #         # # Parameters that we will randomise, generate random values for bar radius, color, and rotation
        #         # bar_radius_range = [10, 30]  # You can adjust the range as needed
        #         # bar_radius = random.uniform(*bar_radius_range)

        #         # # Random color
        #         # random_color = [random.randint(0, 255) for _ in range(3)]
        #         # mask_color = random_color
        #         # rotate_range = [-10, 20]  # Rotation range in degrees
        #         # rotate = math.radians(random.uniform(*rotate_range))

        #         # add the parameters that you want to randomise
        #         img = img
        #         print ("print image path and size")
        #         print(img)
        #         print(img.size)
        #         # Parameters that you randomize
        #         bar_radius_range = [10, 30]
        #         bar_radius = random.uniform(*bar_radius_range)

        #         random_color = [random.randint(0, 255) for _ in range(3)]
        #         mask_color = random_color

        #         rotate_range = [-10, 10]
        #         rotate = math.radians(random.uniform(rotate_range[0], rotate_range[1]))
                
        #         # try:
        #         #     rotate = math.radians(random.uniform(*rotate_range)) 
        #         # except ValueError:
        #         # # retry with new value
        #         #     rotate = math.radians(random.uniform(*rotate_range))

        #         # Create mask
        #         bar_pos_x = [int(i) for i in np.linspace(0, self.resolution[0], self.num_bars_x)]
        #         bar_pos_y = [int(i) for i in np.linspace(0, self.resolution[1], self.num_bars_y)]
        #         bar_locs_x = []
        #         bar_locs_y = []
        #         for i in bar_pos_x:
        #             # int will round down, could use ceil or round
        #             bar_locs_x.extend([j for j in range(int(max(0, i - bar_radius)), int(min(i + bar_radius, self.resolution[0])))])
        #         for i in bar_pos_y:
        #             bar_locs_y.extend([j for j in range(int(max(0, i - bar_radius)), int(min(i + bar_radius, self.resolution[1])))])
        #         mask = np.ones(self.resolution)
        #         for i in range(self.resolution[0]):
        #             for j in range(self.resolution[1]):
        #                 if i in bar_locs_x or j in bar_locs_y:
        #                     mask[i, j] = 0
        #                     mask = cv2.blur(mask, self.kernel, cv2.BORDER_DEFAULT)
        #                     mask = np.array(Image.fromarray(mask).rotate(rotate))

        #         # taking a center crop to remove the rotation artifacts
        #         # plt.imshow(np.array(mask))
        #         crop_size = rotatedRectWithMaxArea(mask.shape[0], mask.shape[1], rotate)
        #         x_boundary = int((mask.shape[0]-crop_size[0])/2)
        #         y_boundary = int((mask.shape[1]-crop_size[1])/2)
        #         mask = mask[x_boundary:-x_boundary, y_boundary:-y_boundary]

        #         # resize and tile
        #         mask = cv2.resize(mask, img.size)
        #         np.expand_dims(mask, 2).shape
        #         np.tile(np.expand_dims(mask, 2), (1,1,3)).shape
        #         tiled_mask = np.tile(np.expand_dims(mask, 2), (1,1,3))

        #         #add bar color
        #         inv_mask = 1-tiled_mask
        #         color = [self.color_alpha * i for i in mask_color]
        #         color_mask = color * inv_mask
        #         new_img = img*tiled_mask+color_mask
        #         return new_img 

    #     # https://stackoverflow.com/questions/76064717/pytorch-resize-specific-dimension-while-keeping-aspect-ratio
    #     # for half of the resize function

        self.transform = Compose([
            FixedHeightResize(224),
            #CageAugmenter(),
            # GaussianBlur(7),
            RandomHorizontalFlip(p=0.5),
            # RandomVerticalFlip (p=0.75),
            # RandomAdjustSharpness(sharpness_factor=5, p=0.5),
            ToTensor(),
        ])
        # the tensor format is channels, height, width

        # RandomShadow
        # CutOut or RandomErasing(p=0.5)
        # GaussianBlur(5)
        
        # index data into list
        self.data = []

        # Determine which root directory to use for annotations based on the split
        if self.split == 'test' and self.test_root:
            annoPath = os.path.join(self.test_root, self.split + '.json')
        else:
            annoPath = os.path.join(self.data_root, self.split + '.json')

        print(annoPath)

        meta = json.load(open(annoPath, 'r'))
        # [print(anno) for anno in meta['annotations'] if anno['image_id']==4236]
        # We previously put the breakpoint before and after this point
        images = dict([[i['id'], i['file_name']] for i in meta['images']])          # image id to filename lookup
        # print (images)
        # print (images.keys())

        labels = dict([[c['id'], idx] for idx, c in enumerate(meta['categories'])]) # custom labelclass indices that start at zero
        print ("length of annotations")
        print(len(meta['annotations']))

        # since we're doing classification, we're just taking the first annotation per image and drop the rest
        images_covered = set()      # all those images for which we have already assigned a label
        for anno in meta['annotations']:
            imgID = anno['image_id']
            if imgID == 4286:
                print(anno)
            if imgID in images_covered:
                continue
            # append image-label tuple to data
            imgFileName = images[imgID]
            label = anno['category_id']
            labelIndex = labels[label]
            self.data.append([imgFileName, labelIndex])
            images_covered.add(imgID) # make sure image is only added once to dataset
    

    def __len__(self):
        '''
            Returns the length of the dataset.
        '''
        return len(self.data)

    def __getitem__(self, idx):
        image_name, label = self.data[idx]
        if self.split == 'test' and self.test_root:
            image_path = os.path.join(self.test_root, image_name)
        else:
            image_path = os.path.join(self.data_root, image_name)
        
        try:
            img = Image.open(image_path).convert('RGB')
        except:
            print(image_path)
            pass

        try:
            img_tensor = self.transform(img)
        except Exception as exc:
            print('bad image:')
            print(type(img))
            print(exc)
            img_tensor = None  # Return None if image loading fails
        
        return img_tensor, label