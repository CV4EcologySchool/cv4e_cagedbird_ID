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
from torchvision.transforms import Compose, Resize, ToTensor, Lambda, Pad, functional, RandomHorizontalFlip
from PIL import Image
import math


class CTDataset(Dataset):

    def __init__(self, cfg, split='train'):
        '''
            Constructor. Here, we collect and index the dataset inputs and
            labels.
        '''
        self.data_root = cfg['data_root']
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

        self.transform = Compose([
            FixedHeightResize(224),
            RandomHorizontalFlip(p=0.5),
            ToTensor(),
        ])
        # the tensor format is channels, height, width
        
        # index data into list
        self.data = []

        # git a annotation file
        annoPath = os.path.join(
            self.data_root,
            'high',
            'training_18_08.json' if self.split=='train' else 'val_18_08.json'
        )

        # print(annoPath)

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
            images_covered.add(imgID)       # make sure image is only added once to dataset
    

    def __len__(self):
        '''
            Returns the length of the dataset.
        '''
        return len(self.data)

    
    def __getitem__(self, idx):
        '''
            Returns a single data point at given idx.
            Here's where we actually load the image.
        '''
        image_name, label = self.data[idx] # see line 57 above where we added these two items to the self.data list
        # load image
        image_path = os.path.join(self.data_root, 'high', image_name)
        # print (image_path)
        try:
            img = Image.open(image_path).convert('RGB') # the ".convert" makes sure we always get three bands in Red, Green, Blue order
        except:
            print(image_path)
            pass # Doesn't do anything if it can't be opened
    
        # print(img.size)
        # transform: see lines 31ff above where we define our transformations
        try:
            img_tensor = self.transform(img)
        except Exception as exc:
            print('bad image:')
            print(type(img))
            print(image_path)
            print(idx)
            raise Exception from exc

        # print(img_tensor.size())
        return img_tensor, label