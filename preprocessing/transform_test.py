import os
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor

data_root = '/home/sicily/cv4e_cagedbird_ID/data'

transform = Compose([              # Transforms. Here's where we could add data augmentation (see Björn's lecture on August 11).
            Resize((cfg['image_size'])),        # For now, we just resize the images to the same dimensions...
            ToTensor()                          # ...and convert them to torch.Tensor.
        ])

image_path = os.path.join(data_root, 'high', 'japanese_grosbeak/japanese_grosbeak_random_190.jpg')
print (image_path)
img = Image.open(image_path).convert('RGB')     # the ".convert" makes sure we always get three bands in Red, Green, Blue order
# print(img.size)
# transform: see lines 31ff above where we define our transformations
img_tensor = transform(img)
print(img_tensor.size())
print(img_tensor, label)