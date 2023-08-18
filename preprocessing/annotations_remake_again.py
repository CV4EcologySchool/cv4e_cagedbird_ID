import os
import json
import random
from PIL import Image

root_directory = "/home/sicily/cv4e_cagedbird_ID/data/high"
output_train_json_path = "/home/sicily/cv4e_cagedbird_ID/data/high/training.json"

# Load COCO annotations from a JSON file
with open(os.path.join(root_directory, "annotations.json"), 'r') as coco_file:
    coco_annotations = json.load(coco_file)

# Shuffle the images and annotations together
combined_data = list(zip(coco_annotations["images"], coco_annotations["annotations"]))
random.shuffle(combined_data)
shuffled_images, shuffled_annotations = zip(*combined_data)

# Calculate the index to split at (80% of the data)
split_index = int(len(shuffled_images) * 0.8)

# Create training dataset
training_data = {
    "images": shuffled_images[:split_index],
    "categories": coco_annotations["categories"],
    "annotations": shuffled_annotations[:split_index]
}

# Create a mapping of old category IDs to new category IDs
category_id_mapping = {category["id"]: idx for idx, category in enumerate(coco_annotations["categories"])}

# Update annotation category IDs to match shuffled annotations
for annotation in training_data["annotations"]:
    old_category_id = annotation["category_id"]
    annotation["category_id"] = category_id_mapping[old_category_id]

# Save training data to a new JSON file
with open(output_train_json_path, 'w') as output_train_json_file:
    json.dump(training_data, output_train_json_file)

print("Training data saved to:", output_train_json_path)
