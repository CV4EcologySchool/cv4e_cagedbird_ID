import os
import json
import random
from PIL import Image
import pickle
from collections import Counter
import numpy as np


root_directory = "/home/sicily/cv4e_cagedbird_ID/data/high"
output_train_json_path = "/home/sicily/cv4e_cagedbird_ID/data/high/training_18_08.json"
output_val_json_path = "/home/sicily/cv4e_cagedbird_ID/data/high/val_18_08.json"
output_upsampling_json_path = "/home/sicily/cv4e_cagedbird_ID/data/high/upsampling.json"

# Load COCO annotations from a JSON file
with open(os.path.join(root_directory, "annotations_test.json"), 'r') as coco_file:
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



# make a JSON to upsample the rarer classes / Make where I will store this .json

# Create the subset training data in a list or a dictionary
# Create upsampled training data


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

# Save upsampled data to a new JSON file
with open(output_upsampling_json_path, 'w') as upsampling_json_file:
    json.dump(upsampled_training_data, upsampling_json_file)

class_mapping = {} # because it is a dictionary

for idx, item in enumerate(training_data["categories"]):
    item["id"]
    item["name"]
    class_mapping[item["id"]] = item["name"]  # Keys need to be strings and there are strings in the name
    class_mapping

with open('./ct_classifier/class_mapping.pickle', 'wb') as f:
    pickle.dump(class_mapping, f)


