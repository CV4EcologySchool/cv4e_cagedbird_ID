import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
import math

# Or just open one image
#load any image
im  = Image.open('/home/sicily/cv4e_cagedbird_ID/data/high/bluethroat/bluethroat_random_31.jpg')

# # Path to the folder containing the images
# folder_path = '/path/to/your/folder'

# # Get a list of image files in the folder
# image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# # Loop through the image files and display each image
# for image_file in image_files:
#     image_path = os.path.join(folder_path, image_file)
    
#     # Load the image using PIL
#     im = Image.open(image_path)

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


#set parameters
resolution = [1000, 1000]
num_bars_x = 4
num_bars_y = 6
kernel = [30, 30]
mask_color = [181, 148, 16]
color_alpha = .3

# Parameters that we will randomise
# Generate random values for bar radius, color, and rotation
bar_radius_range = [10, 30]  # You can adjust the range as needed
bar_radius = random.uniform(*bar_radius_range)

# Random color
random_color = [random.randint(0, 255) for _ in range(3)]
mask_color = random_color
rotate_range = [-20, 20]  # Rotation range in degrees
rotate = math.radians(random.uniform(*rotate_range))

# Insert this code snippet in the appropriate place before creating the mask. This will allow you to generate random 
# values for the bar radius, color, and rotation each time the script is run. Adjust the range values as needed for your desired level of randomness.

#create mask
bar_pos_x = [int(i) for i in np.linspace(0, resolution[0], num_bars_x)]
bar_pos_y = [int(i) for i in np.linspace(0, resolution[1], num_bars_y)]
bar_locs_x = []
bar_locs_y = []
for i in bar_pos_x:
  bar_locs_x.extend([j for j in range(max(0, i - bar_radius), min(i + bar_radius, resolution[0]))])
for i in bar_pos_y:
  bar_locs_y.extend([j for j in range(max(0, i - bar_radius), min(i + bar_radius, resolution[1]))])
mask = np.ones(resolution)
for i in range(resolution[0]):
  for j in range(resolution[1]):
    if i in bar_locs_x or j in bar_locs_y:
      mask[i, j] = 0
mask = cv2.blur(mask, kernel, cv2.BORDER_DEFAULT)
mask = np.array(Image.fromarray(mask).rotate(rotate))

#taking a center crop to remove the rotation artifacts
plt.imshow(np.array(mask))
crop_size = rotatedRectWithMaxArea(mask.shape[0], mask.shape[1], rotate)
x_boundary = int((mask.shape[0]-crop_size[0])/2)
y_boundary = int((mask.shape[1]-crop_size[1])/2)
mask = mask[x_boundary:-x_boundary, y_boundary:-y_boundary]

#resize and tile
mask = cv2.resize(mask, im.size)
np.expand_dims(mask, 2).shape
np.tile(np.expand_dims(mask, 2), (1,1,3)).shape
tiled_mask = np.tile(np.expand_dims(mask, 2), (1,1,3))

#add bar color
inv_mask = 1-tiled_mask
color = [color_alpha * i for i in mask_color]
color_mask = color * inv_mask

#apply to image
plt.imshow((im*tiled_mask+color_mask).astype(np.int64))
plt.axis('off')
# plt.show()
plt.savefig('test_bluethroat.jpg')


padder = Pad(padding)
img = padder(img)
return img
