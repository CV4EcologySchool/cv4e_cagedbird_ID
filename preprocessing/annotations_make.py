import os
import json
from PIL import Image
from pathlib import Path
from glob import glob

# Code needed when you actually regenerate the annotations for the changed dataset
# root_directory = "/home/sicily/cv4e_cagedbird_ID/data/high"
# output_json_path = "/home/sicily/cv4e_cagedbird_ID/data/high/annotations_test.json"

# retest code 01.11.24, briefly change the root directory and run in-terminal, not on ARC4
# Generating annotations for the new 57 species 11.11.24

# root_directory = "/home/home01/bssbf/cv4e_cagedbird_ID/data"
root_directory = "/home/home01/bssbf/cv4e_cagedbird_ID/data2"

output_json_path = "/home/home01/bssbf/cv4e_cagedbird_ID/data2/annotations_test_11_11.json"

# Get a list of category names from subfolder names
category_names = [folder_name for folder_name in os.listdir(root_directory) if os.path.isdir(os.path.join(root_directory, folder_name))]
category_names.sort()

print(len(category_names))
# This should only be 29 species

coco_annotations = {
    "images": [],
    "categories": [],
    "annotations": []
}

# Create category mapping
category_id_mapping = {category_name: idx for idx, category_name in enumerate(category_names)}

# Initialize category IDs
for idx, category_name in enumerate(category_names):
    coco_annotations["categories"].append({
        "id": idx,
        "name": category_name,
        "supercategory": "object"
    })

image_id = 0
annotation_id = 0

for folder in glob (root_directory +'/*'):
    if not Path(folder).is_dir(): # If this folder is a directory or not
        continue 
    for image_path in glob (folder + '/*'):
        if image_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            image = Image.open(image_path)
            image_width, image_height = image.size

            # Determine the category of the image
            category_name = Path (folder).name
            category_id = category_id_mapping[category_name]

            coco_annotations["images"].append({
                "file_name": f'{category_name}/{Path(image_path).name}',
                "height": image_height,
                "width": image_width,
                "id": image_id
            })

            # Create annotation entry for this image
            annotation = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_id,
                "iscrowd": 0,
                "segmentation": [],
                "area": image_width * image_height
            }
            coco_annotations["annotations"].append(annotation)

            image_id += 1
            annotation_id += 1

#     break 
# exit # kills the whole python script
with open(output_json_path, 'w') as output_json_file:
    json.dump(coco_annotations, output_json_file)