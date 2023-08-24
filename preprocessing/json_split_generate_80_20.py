import os
import json
import random
from PIL import Image

root_directory = "/home/sicily/cv4e_cagedbird_ID/data/high"
output_train_json_path = "/home/sicily/cv4e_cagedbird_ID/data/high/training_18_08.json"
output_val_json_path = "/home/sicily/cv4e_cagedbird_ID/data/high/val_18_08.json"

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

# train_data = shuffled_images[:split_index]
# validation_test_data = shuffled_images[split_index:]


# Create a mapping of old category IDs to new category IDs
# category_id_mapping = {category["id"]: idx for idx, category in enumerate(coco_annotations["categories"])}


# Update annotation category IDs to match shuffled annotations
# shuffled_annotations = list(shuffled_annotations)  # Convert to list to modify
# for annotation in shuffled_annotations:
#     old_category_id = annotation["category_id"]
#     annotation["category_id"] = category_id_mapping[old_category_id]

# Create training dataset
training_data = {
    "images": shuffled_images[:split_index],
    "categories": coco_annotations["categories"],  # Include categories here
    "annotations": shuffled_annotations[:split_index]
}

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

for key in training_data["categories"]:
    print(key)

# saved_keys = []  # Create an empty list to store the keys

# for key in training_data["categories"]:
#     print(key)
#     saved_keys.append(key)  # Append the key to the list

# # Save the list to a file
# with open("saved_keys.txt", "w") as file:
#     for key in saved_keys:
#         file.write(key + "\n")

# # Now you have all the keys saved in the 'saved_keys' list


for key in validation_data["categories"]:
    print(key)


