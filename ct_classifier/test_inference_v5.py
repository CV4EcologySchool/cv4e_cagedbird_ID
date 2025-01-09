import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import yaml
import pickle
import csv
import math
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, precision_recall_fscore_support, confusion_matrix, precision_recall_curve, average_precision_score, accuracy_score
from train_save_epoch import create_dataloader, load_model
from torch.utils.data import DataLoader, Dataset
from dataset import CTDataset  # FixedHeightResize is a class in CTDataset
from util import *

# Define FixedHeightResize transformation
class FixedHeightResize:
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        w, h = img.size
        aspect_ratio = float(h) / float(w)
        if h > w:
            new_w = math.ceil(self.size / aspect_ratio)
            img = transforms.functional.resize(img, (self.size, new_w))
        else:
            new_h = math.ceil(aspect_ratio * self.size)
            img = transforms.functional.resize(img, (new_h, self.size))

        w, h = img.size
        pad_diff_h = self.size - h
        pad_diff_w = self.size - w
        padding = [0, pad_diff_h, pad_diff_w, 0]
        padder = transforms.Pad(padding)
        img = padder(img)

        return img

# Parameters, config has been edited to include a test root
config = '/home/home01/bssbf/cv4e_cagedbird_ID/all_model_states/ep100_56sp_ahorflip0.5_lr1e-2_s10/config_ep100_56sp_ahorflip0.5_lr1e-2_s10.yaml'

# Load config
print(f'Using config "{config}"')
cfg = yaml.safe_load(open(config, 'r'))
init_seed(cfg.get('seed', None))

# Load model
model, start_epoch = load_model(cfg, load_latest_version=True)
print(f"Resuming from epoch {start_epoch}")
model.eval()  # Set the model to evaluation mode

# Setup dataset and dataloader for the flat directory structure
class FlatDirectoryDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)

        return image, img_name  # Return image and filename (no label)

# Apply FixedHeightResize transformation
transform = transforms.Compose([
    FixedHeightResize(224),  # Resize with padding to a fixed height of 224
    transforms.ToTensor(),
])

# Set up the DataLoader
image_dir = '/home/home01/bssbf/cv4e_cagedbird_ID/test_con'  # Path to the folder containing all the images
dataset = FlatDirectoryDataset(image_dir=image_dir, transform=transform)
dl_test = DataLoader(dataset, batch_size=128, shuffle=False)  # No shuffling for test set

# Visualize and save a sample of images
sample_batch = next(iter(dl_test))
inputs, filenames = sample_batch

# Visualize and save the images
fig = plt.figure(figsize=(12, 8))
for idx in range(12):  # Display 12 images
    ax = fig.add_subplot(3, 4, idx + 1, xticks=[], yticks=[])
    ax.imshow(inputs[idx].permute(1, 2, 0))  # Permute to (H, W, C)
    ax.set_title(f"Filename: {filenames[idx]}")

plt.tight_layout()
plt.savefig("test_loader_sample2.png")  # Save the plot as PNG
plt.close() 

# Read true labels from the CSV file
true_labels = {}
true_labels_csv = 'true_labels_test2.csv'  # Your true labels CSV file, labels in number format

with open(true_labels_csv, mode='r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip header row
    for row in reader:
        filename, label = row
        true_labels[filename] = int(label)  # Store filenames and their corresponding true labels

# Load class mapping pickle file
class_mapping_file = '/home/home01/bssbf/cv4e_cagedbird_ID/ct_classifier/class_mapping_56.pickle'  # Correct path to your class mapping pickle file

# Load the class mapping pickle file
with open(class_mapping_file, 'rb') as f:
    class_mapping = pickle.load(f)

# **DEBUGGING: Check class mapping structure**
print(f"Class mapping: {class_mapping}")

# Evaluate the model on the unannotated test set
all_preds = []
all_filenames = []
top_2_predictions = []  # To store the top 2 predictions and their confidences

top_1_correct = 0
top_2_correct = 0
total = 0

with torch.no_grad():
    for images, filenames in dl_test:  # Assuming filenames are returned in the DataLoader
        images = images.to(cfg['device'])

        # Make predictions
        outputs = model(images)
        probs = torch.nn.functional.softmax(outputs, dim=1)  # Get probabilities (confidence)

        # Get the top 2 predicted classes and their confidence values
        top2_confidences, top2_preds = torch.topk(probs, 2, dim=1)

        # Collect top 2 predictions, filenames, and confidence values
        for i in range(len(filenames)):
            filename = filenames[i]
            top1_class_index = top2_preds[i][0].item()
            top1_confidence = top2_confidences[i][0].item()
            top2_class_index = top2_preds[i][1].item()
            top2_confidence = top2_confidences[i][1].item()

            # Directly map indices from class_mapping (instead of inverted mapping)
            top1_class = class_mapping.get(top1_class_index, 'Unknown')
            top2_class = class_mapping.get(top2_class_index, 'Unknown')

            top_2_predictions.append([filename, top1_class, top1_confidence, top2_class, top2_confidence])

            # Debugging: Print filenames and mapped classes to check
            print(f"Filename: {filename}, Top 1 Class: {top1_class} (Index: {top1_class_index}), Top 2 Class: {top2_class} (Index: {top2_class_index})")

            true_label = true_labels.get(filename)  # Get the true label for the image
            if true_label is not None:  # Check if true label exists
                total += 1
                # Top 1 Accuracy
                if top1_class == class_mapping.get(true_label, 'Unknown'):
                    top_1_correct += 1
                # Top 2 Accuracy
                if top2_class == class_mapping.get(true_label, 'Unknown') or top1_class == class_mapping.get(true_label, 'Unknown'):
                    top_2_correct += 1

# Calculate Top 1 and Top 2 accuracies
top_1_accuracy = top_1_correct / total if total > 0 else 0
top_2_accuracy = top_2_correct / total if total > 0 else 0

print(f'Top 1 Accuracy: {top_1_accuracy * 100:.2f}%')
print(f'Top 2 Accuracy: {top_2_accuracy * 100:.2f}%')

# Save top 2 predictions with confidence to a CSV
with open("test_pred_top_2_with_confidence2.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(['Filename', 'Top 1 Predicted Class', 'Top 1 Confidence', 'Top 2 Predicted Class', 'Top 2 Confidence'])  # Added confidence
    writer.writerows(top_2_predictions)

print("Top 2 predictions and confidences saved")

import csv
from PIL import Image
import os

# Step 1: Load the average pixel count per class from the CSV
avg_pixel_count_per_class = {}

# Read the CSV file with average pixel counts
with open('/home/home01/bssbf/cv4e_cagedbird_ID/preprocessing/species_by_average_pixel_count.csv', mode='r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip header
    for row in reader:
        species = row[1]  # The folder (species) name is in the 2nd column (index 1)
        avg_pixels = float(row[3])  # The average pixel count is in the 4th column (index 3)
        avg_pixel_count_per_class[species] = avg_pixels

# Step 2: Check the pixel count for each test image and compare it to the class's average
def get_image_resolution(image_path):
    """Get the resolution (width * height) of an image."""
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            return width * height
    except Exception as e:
        print(f"Error reading image {image_path}: {e}")
        return 0

# Step 3: Add this information to your predictions CSV
# We assume that your `true_labels` dictionary and `class_mapping` are already available from the previous steps.

with open("test_pred_top_2_with_confidence2.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        'Filename', 
        'Top 1 Predicted Class', 
        'Top 1 Confidence', 
        'Top 2 Predicted Class', 
        'Top 2 Confidence', 
        'Lower Than Avg Pixel Count'
    ])  # Added new column for lower-than-average pixel count

    for filename, top1_class, top1_confidence, top2_class, top2_confidence in top_2_predictions:
        true_label = true_labels.get(filename)  # Get the true label for the image

        # Retrieve the predicted class name from the class mapping
        predicted_class_name = class_mapping.get(true_label, 'Unknown')

        # Check if the predicted class exists in the pixel count data
        if predicted_class_name in avg_pixel_count_per_class:
            # Get the average pixel count for the predicted class
            avg_pixel_count = avg_pixel_count_per_class[predicted_class_name]
        else:
            avg_pixel_count = 0

        # Step 2: Compare the pixel count of the current image to the class's average
        image_path = os.path.join(image_dir, filename)  # Path to the image file
        image_resolution = get_image_resolution(image_path)  # Get pixel count for the image
        
        # Compare with the average pixel count for the class
        lower_than_avg_pixel_count = image_resolution < avg_pixel_count

        # Write the row to the CSV file
        writer.writerow([
            filename, 
            top1_class, 
            top1_confidence, 
            top2_class, 
            top2_confidence, 
            lower_than_avg_pixel_count  # Add True/False based on the comparison
        ])

print("Top 2 predictions with confidence and pixel count comparison saved.")
