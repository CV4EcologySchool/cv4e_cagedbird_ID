import os
import json
import random
from PIL import Image
import pickle
from collections import Counter
import numpy as np
import math

# root_directory = "/home/sicily/cv4e_cagedbird_ID/data/high"
# output_train_json_path = "/home/sicily/cv4e_cagedbird_ID/data/high/train.json"
# output_upsampling_path = "/home/sicily/cv4e_cagedbird_ID/data/high/upsampling.json"
# output_val_json_path = "/home/sicily/cv4e_cagedbird_ID/data/high/val.json"

# Updated paths from ARC4 not from the node for the summer school
root_directory = "/home/home01/bssbf/cv4e_cagedbird_ID/data2"
output_train_json_path = "/home/home01/bssbf/cv4e_cagedbird_ID/data2/train.json"
output_upsampling_path = "/home/home01/bssbf/cv4e_cagedbird_ID/data2/upsampling.json"
output_val_json_path = "/home/home01/bssbf/cv4e_cagedbird_ID/data2/val.json"


# Load COCO annotations from a JSON file
# with open(os.path.join(root_directory, "annotations_test.json"), 'r') as coco_file:

with open(os.path.join(root_directory, "annotations_test_11_11.json"), 'r') as coco_file:
    coco_annotations = json.load(coco_file)

random.seed (3) # so the shuffle is random but the same each time

# Shuffle the images and annotations together
combined_data = list(zip(coco_annotations["images"], coco_annotations["annotations"]))
random.shuffle(combined_data)
shuffled_images, shuffled_annotations = zip(*combined_data)

# Calculate the index to split at (80% of the data)
split_index = int(len(shuffled_images) * 0.8)

# Create training dataset
training_data = {
    "images": shuffled_images[:split_index],
    "categories": coco_annotations["categories"],  # Include categories here
    "annotations": shuffled_annotations[:split_index]
}

categories = training_data["categories"]
annotations = training_data["annotations"] # because there is only one annotation per image, so I don't need to look at the images too
images = training_data["images"]

# Count the number of images per category
category_image_counts = {category["id"]: 0 for category in training_data["categories"]}
for annotation in training_data["annotations"]:
    category_image_counts[annotation["category_id"]] += 1


# Print summaries for each class
for category_id, count in category_image_counts.items():
    category_name = next(
        (category["name"] for category in training_data["categories"] if category["id"] == category_id),
        f"Category {category_id}",
    )
    print(f"{category_name}: {count} images")

# Find categories with the highest image count
highest_image_count = max(category_image_counts.values())
print("The highest image count for the categories in our dataset")
print (highest_image_count) # The category 'siberian_rubythroat' has the highest image count: 327 images.

# make a JSON to upsample the rarer classes / Make where I will store this .json /  Create upsampled training data
upsampled_training_data = {
    "images": [],
    "categories": training_data["categories"],
    "annotations": []
}

# Now, upsampled_training_data contains the upsampled dataset
for category in training_data["categories"]: 
    category_id = category["id"]
    category_count = category_image_counts[category_id]

    upsample_factor = highest_image_count / category_count
    num_repeats = math.ceil(upsample_factor)
    print (num_repeats)

    annotations_list_for_class = []
    images_list_for_class = []

    for idx, annotation in enumerate(training_data["annotations"]):
        if annotation["category_id"] == category_id:
            annotations_list_for_class.append(annotation)
            corresponding_image = training_data["images"][idx] # so the order should be the same, i.e. the index for each image will match the annotation lsit we maek
            images_list_for_class.append(corresponding_image)

    upsampled_images = np.tile(images_list_for_class, num_repeats)[:highest_image_count]
    upsampled_annotations = np.tile(annotations_list_for_class, num_repeats)[:highest_image_count]
    upsampled_training_data["images"].extend(upsampled_images)
    upsampled_training_data["annotations"].extend(upsampled_annotations)

# use this to check the number of categories after making your list
# Count the number of images per category
category_image_counts_upsampling = {category["id"]: 0 for category in upsampled_training_data["categories"]}
for annotation in upsampled_training_data["annotations"]:
    category_image_counts_upsampling[annotation["category_id"]] += 1

# Print summaries for each class
for category_id, count in category_image_counts_upsampling.items():
    category_name = next(
        (category["name"] for category in upsampled_training_data["categories"] if category["id"] == category_id),
        f"Category {category_id}",
    )
    print(f"{category_name}: {count} images")

# Create validation dataset
validation_data = {
    "images": shuffled_images[split_index:],
    "categories": coco_annotations["categories"],  # Include categories here
    "annotations": shuffled_annotations[split_index:]
}

# Save training data to a new JSON file
with open(output_train_json_path, 'w') as output_train_json_file:
    json.dump(training_data, output_train_json_file)

# Save validation data to a new JSON file
with open(output_val_json_path, 'w') as output_validation_json_file:
    json.dump(validation_data, output_validation_json_file)

# Save upsampling data to a new JSON file
with open(output_upsampling_path, 'w') as output_upsampling_json_file:
    json.dump(upsampled_training_data, output_upsampling_json_file)

class_mapping = {} # because it is a dictionary

for idx, item in enumerate(training_data["categories"]):
    item["id"]
    item["name"]
    class_mapping[item["id"]] = item["name"]  # Keys need to be strings and there are strings in the name
    class_mapping

with open('./ct_classifier/class_mapping.pickle', 'wb') as f:
    pickle.dump(class_mapping, f)
