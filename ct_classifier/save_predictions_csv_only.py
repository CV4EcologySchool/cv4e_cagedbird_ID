import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import yaml
import pickle
import csv
import torch.nn.functional as F  # For softmax function
from train import create_dataloader, load_model
from util import *  # Import the init_seed function

# Parameters
config = '/home/home01/bssbf/cv4e_cagedbird_ID/all_model_states/a-resnet18_d-checked_b-128_n-50_padded_images_sharpness_medium_more_data_upsampled/config_a-resnet18_d-checked_b-128_n-50_padded_images_sharpness_medium_more_data_upsampled.yaml'
split = 'val'

# Load config
print(f'Using config "{config}"')
cfg = yaml.safe_load(open(config, 'r'))

# Set the seed, so you can reproduce the randomness, None is there as null because the seed is already in the config
init_seed(cfg.get('seed', None))

# Setup dataloader
dl_val = create_dataloader(cfg, split='val')  # Get the validation dataloader

# Load model
model, start_epoch = load_model(cfg, load_latest_version=True)
print(start_epoch)

# Predict and save results
inputs_list = []
labels_list = []
pred_list = []
confidence_score_list = []  # List to store the confidence score for each prediction
mismatch_list = []  # List to store mismatch info

# Load the class mapping dictionary (if it's available as a pickle file)
class_mapping_file = '/home/home01/bssbf/cv4e_cagedbird_ID/ct_classifier/class_mapping_56.pickle'
with open(class_mapping_file, 'rb') as f:
    class_mapping = pickle.load(f)

print(class_mapping)

# Create a directory to save the images
output_dir = 'predicted_images'
os.makedirs(output_dir, exist_ok=True)

# Iterate over validation data
for batch_idx, (inputs, labels) in enumerate(dl_val):
    predictions = model(inputs)  # Forward pass
    probabilities = F.softmax(predictions, dim=1)  # Apply softmax to get probabilities
    max_pred, argmax_pred = probabilities.max(dim=1)  # Get predicted class index and max probability (confidence)
    
    # For each prediction, check if it matches the true label and store accuracy (1 for correct, 0 for incorrect)
    for idx, (pred, true, score) in enumerate(zip(argmax_pred, labels, max_pred)):
        accuracy = 1 if pred == true else 0  # 1 for correct, 0 for incorrect
        confidence_score_list.append(score.item())  # Store the confidence score (probability between 0 and 1)
        mismatch_list.append('Mismatch' if pred != true else 'Match')  # Track mismatches
        
        # Save the image with the predicted label
        img = inputs[idx].permute(1, 2, 0).cpu().numpy()  # Convert tensor to numpy array
        plt.imshow(img)
        plt.title(f'Predicted: {class_mapping.get(pred.item(), pred.item())}, True: {class_mapping.get(true.item(), true.item())}')
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, f'batch{batch_idx}_img{idx}.png'))
        plt.close()
    
    # Extend the lists for predictions and true labels
    pred_list.extend(list(argmax_pred))
    inputs_list.extend(list(inputs))
    labels_list.extend(list(labels))

# Map the predicted class indices to class labels
predicted_labels = [class_mapping.get(pred.item(), pred.item()) for pred in pred_list]
true_labels = [class_mapping.get(label.item(), label.item()) for label in labels_list]

# Save the predictions and true labels to CSV
with open('validation_predictions5.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['True Label', 'Predicted Label', 'Confidence Score', 'Mismatch'])  # Write header
    for true_label, pred_label, score, mismatch in zip(true_labels, predicted_labels, confidence_score_list, mismatch_list):
        writer.writerow([true_label, pred_label, score, mismatch])  # Save results