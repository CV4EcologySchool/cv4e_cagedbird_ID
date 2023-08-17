import os
import json
from PIL import Image

root_directory = "/home/sicily/cv4e_cagedbird_ID/data/high"
output_json_path = "/home/sicily/cv4e_cagedbird_ID/data/high/annotations.json"

# Get a list of category names from subfolder names
category_names = [folder_name for folder_name in os.listdir(root_directory) if os.path.isdir(os.path.join(root_directory, folder_name))]
category_names.sort()

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

for dir_name, _, files in os.walk(root_directory):
    for file_name in files:
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            image_path = os.path.join(dir_name, file_name)
            image = Image.open(image_path)
            image_width, image_height = image.size

            # Determine the category of the image
            category_name = os.path.basename(dir_name)
            category_id = category_id_mapping[category_name]

            coco_annotations["images"].append({
                "file_name": file_name,
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

with open(output_json_path, 'w') as output_json_file:
    json.dump(coco_annotations, output_json_file)