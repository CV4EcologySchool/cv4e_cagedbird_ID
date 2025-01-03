import json
import random

# Load the annotation.json file
with open('/home/sicily/cv4e_cagedbird_ID/data/high/annotations.json', 'r') as f:
    annotations = json.load(f)

# Initialize dictionaries for training and validation annotations
training_data = {}
validation_data = {}

# Define the split ratio (80% training, 20% validation)
split_ratio = 0.8

# Iterate through each class in the annotation
for class_name, class_annotations in annotations.items():
    total_samples = len(class_annotations)
    split_index = int(total_samples * split_ratio)

    # Randomly shuffle the annotations
    random.shuffle(class_annotations)

    # Split annotations into training and validation
    training_annotations = class_annotations[:split_index]
    validation_annotations = class_annotations[split_index:]

    # Update training_data and validation_data dictionaries
    training_data[class_name] = training_annotations
    validation_data[class_name] = validation_annotations

# Create and write the training.json file
with open('/home/sicily/cv4e_cagedbird_ID/data/high/training.json', 'w') as f:
    json.dump(training_data, f, indent=4)

# Create and write the validation.json file
with open('/home/sicily/cv4e_cagedbird_ID/data/high/validation.json', 'w') as f:
    json.dump(validation_data, f, indent=4)

print("Splitting and writing completed.")
